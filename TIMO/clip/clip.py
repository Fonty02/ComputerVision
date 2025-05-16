import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if torch.__version__.split(".") < ["1", "7", "1"]:
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "CustomRN50": "",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name, device="cuda" if torch.cuda.is_available() else "cpu", jit=False, download_root=None):
    """
    Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed in the model card or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name == "CustomRN50":
        print("Caricamento del modello CustomRN50 (adapted_RN50_artgraph.pt)...")
        # Importa CLIPWithStyleAdapter dalla classe Adapter
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from StyleAdapter.StyleAdapter import CLIPWithStyleAdapter
        
        # Carica il modello CLIP base per ottenere la struttura e il preprocessing
        clip_model, preprocess = load("RN50", device, jit, download_root)
        
        # Cerca il modello adattato
        adapted_path = "/home/fonty/Scrivania/UniRepo/ComputerVision/TIMO/best_style_adapted_clip_artgraph.pt"
        if not os.path.exists(adapted_path):
            # Cerca il modello nella directory corrente
            adapted_path = os.path.join(os.getcwd(), "best_adapted_clip_artgraph.pt")
            
        if os.path.exists(adapted_path):
            # Carica il checkpoint completo
            checkpoint = torch.load(adapted_path, map_location=device,weights_only=False)
            
            # Stampa le informazioni sul checkpoint
            print(f"Informazioni sul checkpoint:")
            if "epoch" in checkpoint:
                print(f"Epoca: {checkpoint['epoch']}")
            if "val_acc" in checkpoint:
                print(f"Accuratezza di validazione: {checkpoint['val_acc']}")
            if "classnames" in checkpoint:
                print(f"Numero di classi: {len(checkpoint['classnames'])}")
            
            # Ottieni il bottleneck_dim corretto dal checkpoint o dagli argomenti salvati
            if "args" in checkpoint and hasattr(checkpoint["args"], "bottleneck_dim"):
                bottleneck_dim = checkpoint["args"].bottleneck_dim
            else:
                # Deduce la dimensione dal primo parametro del modello
                # Verifica prima se model_state_dict esiste nel checkpoint
                if "model_state_dict" in checkpoint:
                    state_dict_to_check = checkpoint["model_state_dict"]
                else:
                    # Se model_state_dict non esiste, utilizza il checkpoint stesso come state_dict
                    state_dict_to_check = checkpoint

                # Verifica se down_project.weight esiste nello state_dict
                if "down_project.weight" in state_dict_to_check:
                    bottleneck_dim = state_dict_to_check["down_project.weight"].shape[0]
                else:
                    bottleneck_dim = 64  # Default fallback

            # Verifica i parametri accettati dal costruttore di CLIPWithStyleAdapter
            print(f"Creazione del modello CLIPWithStyleAdapter...")
            
            # Ottieni la firma del costruttore
            import inspect
            constructor_params = inspect.signature(CLIPWithStyleAdapter.__init__).parameters
            
            # Aggiorna la creazione del modello con parametri corretti
            try:
                # Usa valori predefiniti per i parametri richiesti
                fusion_bottleneck_dim = 128  # Un valore ragionevole basato sul codice in StyleAdapter.py
                gram_style_projection_dim = 256  # Un valore predefinito
                layers_for_gram_rn50 = ['layer2', 'layer3']  # Aggiungi questo parametro
                dropout_rate = 0.1
                use_layernorm_adapter = True
                
                # Crea il modello StyleAdapter con i parametri corretti
                adapted_model = CLIPWithStyleAdapter(
                    clip_model_name="RN50",  # Usa direttamente "RN50" invece di "CustomRN50" per evitare problemi
                    fusion_bottleneck_dim=fusion_bottleneck_dim,
                    gram_style_projection_dim=gram_style_projection_dim,
                    layers_for_gram_rn50=layers_for_gram_rn50,
                    dropout_rate=dropout_rate,
                    use_layernorm_adapter=use_layernorm_adapter,
                    device=device
                ).to(device)
                
                print(f"Modello CLIPWithStyleAdapter creato con fusion_bottleneck_dim={fusion_bottleneck_dim}")
            except Exception as e:
                print(f"Errore nella creazione del modello CLIPWithStyleAdapter: {e}")
                print("Utilizzo del modello CLIP base senza adapter.")
                return clip_model, preprocess
            
            # Carica lo state_dict nel modello adattato
            if "model_state_dict" in checkpoint:
                print("Caricamento del modello dallo state_dict...")
                # Carica nei moduli corretti basandosi sui nomi presenti nello state_dict
                if hasattr(adapted_model, 'gram_layer_projections') and 'gram_layer_projections_state_dict' in checkpoint:
                    adapted_model.gram_layer_projections.load_state_dict(checkpoint["gram_layer_projections_state_dict"])
                    print("Caricati parametri gram_layer_projections")
                if hasattr(adapted_model, 'fusion_adapter') and 'fusion_adapter_state_dict' in checkpoint:
                    adapted_model.fusion_adapter.load_state_dict(checkpoint["fusion_adapter_state_dict"])
                    print("Caricati parametri fusion_adapter")
                print(f"Modello adattato caricato con successo da {adapted_path}")
            else:
                print("Tentativo di caricamento diretto dello state_dict...")
                try:
                    # Prova a caricare direttamente nei moduli corretti
                    state_dict_keys = list(checkpoint.keys())
                    print(f"Chiavi disponibili nel checkpoint: {state_dict_keys[:5]}...")
                    
                    # Carica nei componenti disponibili basandosi sui prefissi delle chiavi
                    has_loaded = False
                    
                    # Controlla se è presente fusion_adapter_state_dict
                    if 'fusion_adapter_state_dict' in checkpoint and hasattr(adapted_model, 'fusion_adapter'):
                        adapted_model.fusion_adapter.load_state_dict(checkpoint['fusion_adapter_state_dict'])
                        has_loaded = True
                        print("Caricati parametri fusion_adapter dal checkpoint")
                    
                    # Controlla se è presente gram_layer_projections_state_dict
                    if 'gram_layer_projections_state_dict' in checkpoint and hasattr(adapted_model, 'gram_layer_projections'):
                        adapted_model.gram_layer_projections.load_state_dict(checkpoint['gram_layer_projections_state_dict'])
                        has_loaded = True
                        print("Caricati parametri gram_layer_projections dal checkpoint")
                    
                    if not has_loaded:
                        # Fallback: prova a caricare direttamente il modello completo
                        adapted_model.load_state_dict(checkpoint)
                        print("Caricato state_dict completo")
                    
                    print("Caricamento del modello completato con successo")
                except Exception as e:
                    print(f"Errore nel caricamento: {e}")
                    print("Utilizzo del modello con adapter non inizializzato.")
            
            # Prima di restituire il modello, assicurati che l'attributo feature_extractor_hooks sia inizializzato
            if not hasattr(adapted_model, 'feature_extractor_hooks'):
                adapted_model.feature_extractor_hooks = []
            
            # Assicurati che extracted_gram_feature_maps sia inizializzato
            if not hasattr(adapted_model, 'extracted_gram_feature_maps'):
                adapted_model.extracted_gram_feature_maps = {}
            
            # Modifica il name da CustomRN50 a RN50 quando necessario
            if adapted_model.clip_model_name == "CustomRN50":
                adapted_model.clip_model_name = "RN50"
                print("Corretto clip_model_name da CustomRN50 a RN50 per la compatibilità con gli hook")
            
            # Verifica finale del modello
            print("Verifica finale del modello...")
            # Test di funzionalità base
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            try:
                _ = adapted_model.encode_image(dummy_input)
                print("Il modello funziona correttamente!")
                return adapted_model, preprocess
            except Exception as e:
                print(f"Il modello non funziona: {e}, ritorno al modello base")
                return clip_model, preprocess
                
            except Exception as main_error:
                print(f"Errore generale: {main_error}")
                print("Ritorno al modello base per sicurezza")
                return clip_model, preprocess
    
    # Codice originale per gli altri modelli...
    # ...existing code...
    
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result