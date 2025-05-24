import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph
import os
import sys
import argparse

def load_model(model_path):
    """
    Carica un modello PyTorch da file .pt gestendo i nuovi requisiti di sicurezza
    """
    try:
        print("Tentativo di caricamento con weights_only=False...")
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if isinstance(model, dict):
            if 'args' in model or any('adapter' in key for key in model.keys()):
                return load_custom_model_from_checkpoint(model, model_path)
            else:
                return None
        
        model.eval()
        return model
        
    except Exception as e:
        try:
            torch.serialization.add_safe_globals([argparse.Namespace])
            model = torch.load(model_path, map_location='cpu', weights_only=True)
            return model
        except Exception as e2:
            return None

def load_custom_model_from_checkpoint(checkpoint, model_path):
    """
    Ricostruisce il modello CLIPWithStyleAdapter dal checkpoint
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        style_adapter_path = os.path.join(current_dir, 'StyleAdapter')
        
        if style_adapter_path not in sys.path:
            sys.path.append(style_adapter_path)
        
        from StyleAdapterFLS import CLIPWithStyleAdapter
        import clip
        
        fusion_bottleneck_dim = checkpoint.get('fusion_bottleneck_dim', 128)
        args = checkpoint.get('args', None)
        
        if args is None:
            gram_style_projection_dim = 512
            layers_for_gram_rn50 = ['layer4.2.bn3', 'layer4.1.bn3', 'layer4.0.bn3']
            dropout_rate = 0.1
            use_layernorm_adapter = True
        else:
            gram_style_projection_dim = getattr(args, 'gram_style_projection_dim', 512)
            layers_for_gram_rn50 = getattr(args, 'layers_for_gram_rn50', ['layer4.2.bn3', 'layer4.1.bn3', 'layer4.0.bn3'])
            dropout_rate = checkpoint.get('dropout_rate', 0.1)
            use_layernorm_adapter = getattr(args, 'use_layernorm_adapter', True)
        
        model = CLIPWithStyleAdapter(
            clip_model_name="RN50",
            fusion_bottleneck_dim=fusion_bottleneck_dim,
            gram_style_projection_dim=gram_style_projection_dim,
            layers_for_gram_rn50=layers_for_gram_rn50,
            dropout_rate=dropout_rate,
            use_layernorm_adapter=use_layernorm_adapter,
            device='cpu'
        )
        
        if 'fusion_adapter_state_dict' in checkpoint and hasattr(model, 'fusion_adapter'):
            model.fusion_adapter.load_state_dict(checkpoint['fusion_adapter_state_dict'])
        
        if 'gram_layer_projections_state_dict' in checkpoint and hasattr(model, 'gram_layer_projections'):
            model.gram_layer_projections.load_state_dict(checkpoint['gram_layer_projections_state_dict'])
        
        model.eval()
        return model
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def create_detailed_diagram(model, save_path="detailed_model_diagram"):
    """
    Crea un diagramma dettagliato che mostra i componenti specifici dell'architettura
    """
    try:
        dot = Digraph(comment='Detailed Model Structure', format='png')
        # Formato orizzontale per paper - "in lungo"
        dot.attr(rankdir='LR', size="20,8", dpi="300", ratio="fill")
        dot.attr('node', shape='box', style='rounded,filled', fontsize='9', width='1.2', height='0.8')
        dot.attr('edge', fontsize='7', arrowsize='0.7')
        
        # Analizza il modello per identificare i componenti
        has_adapter = False
        has_style = False
        has_gram = False
        gram_layers = []
        
        for name, module in model.named_modules():
            if 'adapter' in name.lower():
                has_adapter = True
            if 'style' in name.lower() or 'gram' in name.lower():
                has_style = True
            if 'gram' in name.lower():
                has_gram = True
                if 'layer4' in name:
                    gram_layers.append(name)
        
        # Input - Prima colonna
        with dot.subgraph(name='cluster_0') as c0:
            c0.attr(rank='same')
            c0.node('input_img', 'Input Image\n(3×224×224)', fillcolor='lightblue')
            c0.node('input_text', 'Input Text\n(tokenized)', fillcolor='lightblue')
        
        # CLIP Encoders - Seconda colonna
        with dot.subgraph(name='cluster_1') as c1:
            c1.attr(rank='same')
            # Visual Encoder compatto
            c1.node('visual_encoder', 'CLIP Visual Encoder\n(ResNet-50)\nStem → L1 → L2 → L3 → L4', 
                    fillcolor='lightcoral', width='2.0', height='1.2')
            c1.node('text_encoder', 'CLIP Text Encoder\nEmbedding → Transformer\n(12 layers)', 
                    fillcolor='lightcoral', width='2.0', height='1.2')
        
        # Feature Extraction - Terza colonna
        with dot.subgraph(name='cluster_2') as c2:
            c2.attr(rank='same')
            c2.node('visual_features', 'Visual Features\n(2048-dim)', fillcolor='orange')
            c2.node('text_features', 'Text Features\n(512-dim)', fillcolor='orange')
            
            if has_gram:
                c2.node('gram_features', 'Gram Matrices\nL4.0, L4.1, L4.2\n(Style Features)', 
                        fillcolor='lightgreen', width='1.5')
        
        # Projections - Quarta colonna
        with dot.subgraph(name='cluster_3') as c3:
            c3.attr(rank='same')
            c3.node('visual_proj', 'Visual Projection\n(2048→1024)', fillcolor='lightyellow')
            c3.node('text_proj', 'Text Projection\n(512→1024)', fillcolor='lightyellow')
            
            if has_gram:
                c3.node('gram_proj', 'Gram Projections\n(Style→512)', fillcolor='lightgreen')
        
        # Fusion/Adaptation - Quinta colonna (se presente)
        if has_adapter:
            with dot.subgraph(name='cluster_4') as c4:
                c4.attr(rank='same')
                c4.node('fusion_adapter', 'Fusion Adapter\nContent + Style\nBottleneck Fusion', 
                        fillcolor='gold', width='1.5', height='1.0')
                c4.node('text_final', 'Text Embeddings\n(1024-dim)', fillcolor='lightpink')
        
        # Output - Ultima colonna
        final_cluster_name = 'cluster_5' if has_adapter else 'cluster_4'
        with dot.subgraph(name=final_cluster_name) as cf:
            cf.attr(rank='same')
            if has_adapter:
                cf.node('visual_final', 'Adapted Visual\nEmbeddings\n(1024-dim)', fillcolor='lightpink')
            else:
                cf.node('visual_final', 'Visual Embeddings\n(1024-dim)', fillcolor='lightpink')
                cf.node('text_final', 'Text Embeddings\n(1024-dim)', fillcolor='lightpink')
            
            cf.node('similarity', 'Cosine\nSimilarity\nScore', fillcolor='mediumpurple', shape='ellipse')
        
        # Connessioni principali - Flusso da sinistra a destra
        # Input → Encoders
        dot.edge('input_img', 'visual_encoder')
        dot.edge('input_text', 'text_encoder')
        
        # Encoders → Features
        dot.edge('visual_encoder', 'visual_features')
        dot.edge('text_encoder', 'text_features')
        
        # Features → Projections
        dot.edge('visual_features', 'visual_proj')
        dot.edge('text_features', 'text_proj')
        
        # Style path (se presente)
        if has_gram:
            dot.edge('visual_encoder', 'gram_features', label='L4 features', 
                    style='dashed', color='green')
            dot.edge('gram_features', 'gram_proj')
        
        # Fusion path (se presente)
        if has_adapter:
            dot.edge('visual_proj', 'fusion_adapter', label='content')
            if has_gram:
                dot.edge('gram_proj', 'fusion_adapter', label='style')
            dot.edge('fusion_adapter', 'visual_final')
            dot.edge('text_proj', 'text_final')
        else:
            dot.edge('visual_proj', 'visual_final')
            dot.edge('text_proj', 'text_final')
            if has_gram:
                dot.edge('gram_proj', 'visual_final', style='dashed', color='green')
        
        # Final similarity
        dot.edge('visual_final', 'similarity')
        dot.edge('text_final', 'similarity')
        
        
        # Salva il diagramma
        dot.render(save_path, cleanup=True)
        print(f"Diagramma salvato come {save_path}.png")
        
    except Exception as e:
        print(f"Errore nella creazione del diagramma: {e}")
        import traceback
        traceback.print_exc()

def main():
    model_path = "fsl4_best_style_adapted_clip_artgraph.pt"
    
    model = load_model(model_path)
    
    if model is None:
        return
    if "fsl1" in model_path:
        save_path = "fsl1_detailed_model_diagram"
    elif "fsl4" in model_path:
        save_path = "fsl4_detailed_model_diagram"
    create_detailed_diagram(model, save_path=save_path)

if __name__ == "__main__":
    main()