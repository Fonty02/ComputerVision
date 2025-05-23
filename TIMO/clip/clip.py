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
    "Custom_FSL_RN50" :"",
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
    if name == "Custom_FSL_RN50":
            print("Caricamento del modello Custom_FSL_RN50 ...")
            import sys
            import json
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            sys.path.append(project_root)

            from StyleAdapter.StyleAdapterFLS import CLIPWithStyleAdapter
            
            # Carica il modello CLIP base per ottenere la struttura e il preprocessing
            clip_model, preprocess = load("RN50", device, jit, download_root)
            
            # Percorsi per il modello e gli iperparametri
            adapted_path = os.path.join(project_root, "fsl4_best_style_adapted_clip_artgraph.pt")
            
            # Verifica se esiste il checkpoint
            if os.path.exists(adapted_path):
                # Carica prima il checkpoint per determinare le dimensioni corrette
                checkpoint = torch.load(adapted_path, map_location=device, weights_only=False)
                
                print(f"Informazioni sul checkpoint:")
                if "epoch" in checkpoint:
                    print(f"Epoca: {checkpoint['epoch']}")
                if "val_acc" in checkpoint:
                    print(f"Accuratezza di validazione: {checkpoint['val_acc']}")
                if "classnames" in checkpoint:
                    print(f"Numero di classi: {len(checkpoint['classnames'])}")
                if "fusion_bottleneck_dim" in checkpoint:
                    print(f"Dimensione bottleneck: {checkpoint['fusion_bottleneck_dim']}")
                
                # Estrai la dimensione del bottleneck dal checkpoint stesso
                # Analisi dei parametri nel checkpoint per determinare la dimensione corretta
                fusion_bottleneck_dim = None
                if "fusion_adapter_state_dict" in checkpoint:
                    try:
                        # Estrai la dimensione direttamente dai pesi
                        down_project_weight = checkpoint["fusion_adapter_state_dict"].get("down_project.weight")
                        if down_project_weight is not None:
                            fusion_bottleneck_dim = down_project_weight.shape[0]
                            print(f"Dimensione bottleneck determinata dai pesi: {fusion_bottleneck_dim}")
                    except Exception as e:
                        print(f"Errore nell'analisi dei pesi: {e}")
                
                # Usa un valore di default solo se non è possibile determinare la dimensione
                if fusion_bottleneck_dim is None:
                    if "fusion_bottleneck_dim" in checkpoint:
                        fusion_bottleneck_dim = checkpoint["fusion_bottleneck_dim"]
                        print(f"Dimensione bottleneck dal checkpoint: {fusion_bottleneck_dim}")
                    else:
                        fusion_bottleneck_dim = 256  # Valore di fallback
                        print(f"Usando dimensione bottleneck di fallback: {fusion_bottleneck_dim}")
                
                # Altri parametri di configurazione
                config_gram_style_projection_dim = 256  # Valore fisso
                config_layers_for_gram_rn50 = ['layer2', 'layer3']  # Valore fisso
                config_dropout_rate = checkpoint.get("dropout_rate", 0.1)  # Usa 0.1 come default
                config_use_layernorm_adapter = True  # Valore fisso
                
                print(f"Creazione modello con parametri: "
                    f"fusion_bottleneck_dim={fusion_bottleneck_dim}, "
                    f"dropout_rate={config_dropout_rate}")
                    
                try:
                    adapted_model = CLIPWithStyleAdapter(
                        clip_model_name="RN50",
                        fusion_bottleneck_dim=fusion_bottleneck_dim,
                        gram_style_projection_dim=config_gram_style_projection_dim,
                        layers_for_gram_rn50=config_layers_for_gram_rn50,
                        dropout_rate=config_dropout_rate,
                        use_layernorm_adapter=config_use_layernorm_adapter,
                        device=device
                    ).to(device)
                except Exception as e:
                    print(f"Errore nella creazione del modello CLIPWithStyleAdapter: {e}")
                    return clip_model, preprocess
                
                # Carica lo state_dict nel modello adattato
                loaded_successfully = False
                if 'fusion_adapter_state_dict' in checkpoint and hasattr(adapted_model, 'fusion_adapter'):
                    try:
                        adapted_model.fusion_adapter.load_state_dict(checkpoint['fusion_adapter_state_dict'])
                        print("Caricati parametri fusion_adapter dal checkpoint.")
                        loaded_successfully = True
                    except RuntimeError as e:
                        print(f"Errore nel caricare fusion_adapter_state_dict: {e}")
                
                if 'gram_projections_state_dict' in checkpoint and hasattr(adapted_model, 'gram_layer_projections'):
                    try:
                        adapted_model.gram_layer_projections.load_state_dict(checkpoint['gram_projections_state_dict'])
                        print("Caricati parametri gram_layer_projections dal checkpoint.")
                        loaded_successfully = True
                    except RuntimeError as e:
                        print(f"Errore nel caricare gram_projections_state_dict: {e}")
                        # Prova con il nome alternativo
                        if 'gram_layer_projections_state_dict' in checkpoint:
                            try:
                                adapted_model.gram_layer_projections.load_state_dict(checkpoint['gram_layer_projections_state_dict'])
                                print("Caricati parametri gram_layer_projections dal checkpoint (nome alternativo).")
                                loaded_successfully = True
                            except RuntimeError as e:
                                print(f"Errore nel caricare gram_layer_projections_state_dict: {e}")
                
                # Assicurati che il modello abbia gli attributi necessari
                if not hasattr(adapted_model, 'feature_extractor_hooks'):
                    adapted_model.feature_extractor_hooks = []
                if not hasattr(adapted_model, 'extracted_gram_feature_maps'):
                    adapted_model.extracted_gram_feature_maps = {}
                
                # Verifica finale del modello
                if loaded_successfully:
                    dummy_input = torch.randn(1, 3, adapted_model.visual.input_resolution, adapted_model.visual.input_resolution).to(device)
                    try:
                        with torch.no_grad():
                            _ = adapted_model.encode_image(dummy_input)
                        print("Il modello Custom_FSL_RN50 funziona correttamente!")
                        return adapted_model, preprocess
                    except Exception as e:
                        print(f"Il modello Custom_FSL_RN50 non funziona dopo il caricamento: {e}")
                else:
                    print("Nessun parametro caricato correttamente dal checkpoint.")
                
                return clip_model, preprocess
            else:
                print(f"File del modello adattato non trovato in {adapted_path}. Ritorno al modello CLIP base.")
                return clip_model, preprocess
            
    if name == "Custom_FSL1_RN50":
            print("Caricamento del modello Custom_FSL_RN50 ...")
            import sys
            import json
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            sys.path.append(project_root)

            from StyleAdapter.StyleAdapterFLS import CLIPWithStyleAdapter
            
            # Carica il modello CLIP base per ottenere la struttura e il preprocessing
            clip_model, preprocess = load("RN50", device, jit, download_root)
            
            # Percorsi per il modello e gli iperparametri
            adapted_path = os.path.join(project_root, "fsl1_best_style_adapted_clip_artgraph.pt")
            
            # Verifica se esiste il checkpoint
            if os.path.exists(adapted_path):
                # Carica prima il checkpoint per determinare le dimensioni corrette
                checkpoint = torch.load(adapted_path, map_location=device, weights_only=False)
                
                print(f"Informazioni sul checkpoint:")
                if "epoch" in checkpoint:
                    print(f"Epoca: {checkpoint['epoch']}")
                if "val_acc" in checkpoint:
                    print(f"Accuratezza di validazione: {checkpoint['val_acc']}")
                if "classnames" in checkpoint:
                    print(f"Numero di classi: {len(checkpoint['classnames'])}")
                if "fusion_bottleneck_dim" in checkpoint:
                    print(f"Dimensione bottleneck: {checkpoint['fusion_bottleneck_dim']}")
                
                # Estrai la dimensione del bottleneck dal checkpoint stesso
                # Analisi dei parametri nel checkpoint per determinare la dimensione corretta
                fusion_bottleneck_dim = None
                if "fusion_adapter_state_dict" in checkpoint:
                    try:
                        # Estrai la dimensione direttamente dai pesi
                        down_project_weight = checkpoint["fusion_adapter_state_dict"].get("down_project.weight")
                        if down_project_weight is not None:
                            fusion_bottleneck_dim = down_project_weight.shape[0]
                            print(f"Dimensione bottleneck determinata dai pesi: {fusion_bottleneck_dim}")
                    except Exception as e:
                        print(f"Errore nell'analisi dei pesi: {e}")
                
                # Usa un valore di default solo se non è possibile determinare la dimensione
                if fusion_bottleneck_dim is None:
                    if "fusion_bottleneck_dim" in checkpoint:
                        fusion_bottleneck_dim = checkpoint["fusion_bottleneck_dim"]
                        print(f"Dimensione bottleneck dal checkpoint: {fusion_bottleneck_dim}")
                    else:
                        fusion_bottleneck_dim = 256  # Valore di fallback
                        print(f"Usando dimensione bottleneck di fallback: {fusion_bottleneck_dim}")
                
                # Altri parametri di configurazione
                config_gram_style_projection_dim = 256  # Valore fisso
                config_layers_for_gram_rn50 = ['layer2', 'layer3']  # Valore fisso
                config_dropout_rate = checkpoint.get("dropout_rate", 0.1)  # Usa 0.1 come default
                config_use_layernorm_adapter = True  # Valore fisso
                
                print(f"Creazione modello con parametri: "
                    f"fusion_bottleneck_dim={fusion_bottleneck_dim}, "
                    f"dropout_rate={config_dropout_rate}")
                    
                try:
                    adapted_model = CLIPWithStyleAdapter(
                        clip_model_name="RN50",
                        fusion_bottleneck_dim=fusion_bottleneck_dim,
                        gram_style_projection_dim=config_gram_style_projection_dim,
                        layers_for_gram_rn50=config_layers_for_gram_rn50,
                        dropout_rate=config_dropout_rate,
                        use_layernorm_adapter=config_use_layernorm_adapter,
                        device=device
                    ).to(device)
                except Exception as e:
                    print(f"Errore nella creazione del modello CLIPWithStyleAdapter: {e}")
                    return clip_model, preprocess
                
                # Carica lo state_dict nel modello adattato
                loaded_successfully = False
                if 'fusion_adapter_state_dict' in checkpoint and hasattr(adapted_model, 'fusion_adapter'):
                    try:
                        adapted_model.fusion_adapter.load_state_dict(checkpoint['fusion_adapter_state_dict'])
                        print("Caricati parametri fusion_adapter dal checkpoint.")
                        loaded_successfully = True
                    except RuntimeError as e:
                        print(f"Errore nel caricare fusion_adapter_state_dict: {e}")
                
                if 'gram_projections_state_dict' in checkpoint and hasattr(adapted_model, 'gram_layer_projections'):
                    try:
                        adapted_model.gram_layer_projections.load_state_dict(checkpoint['gram_projections_state_dict'])
                        print("Caricati parametri gram_layer_projections dal checkpoint.")
                        loaded_successfully = True
                    except RuntimeError as e:
                        print(f"Errore nel caricare gram_projections_state_dict: {e}")
                        # Prova con il nome alternativo
                        if 'gram_layer_projections_state_dict' in checkpoint:
                            try:
                                adapted_model.gram_layer_projections.load_state_dict(checkpoint['gram_layer_projections_state_dict'])
                                print("Caricati parametri gram_layer_projections dal checkpoint (nome alternativo).")
                                loaded_successfully = True
                            except RuntimeError as e:
                                print(f"Errore nel caricare gram_layer_projections_state_dict: {e}")
                
                # Assicurati che il modello abbia gli attributi necessari
                if not hasattr(adapted_model, 'feature_extractor_hooks'):
                    adapted_model.feature_extractor_hooks = []
                if not hasattr(adapted_model, 'extracted_gram_feature_maps'):
                    adapted_model.extracted_gram_feature_maps = {}
                
                # Verifica finale del modello
                if loaded_successfully:
                    dummy_input = torch.randn(1, 3, adapted_model.visual.input_resolution, adapted_model.visual.input_resolution).to(device)
                    try:
                        with torch.no_grad():
                            _ = adapted_model.encode_image(dummy_input)
                        print("Il modello Custom_FSL_RN50 funziona correttamente!")
                        return adapted_model, preprocess
                    except Exception as e:
                        print(f"Il modello Custom_FSL_RN50 non funziona dopo il caricamento: {e}")
                else:
                    print("Nessun parametro caricato correttamente dal checkpoint.")
                
                return clip_model, preprocess
            else:
                print(f"File del modello adattato non trovato in {adapted_path}. Ritorno al modello CLIP base.")
                return clip_model, preprocess
        
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