import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import random
import numpy as np

def load_sample_images(dataset_path, num_samples=6):
    """
    Carica un campione di immagini dal dataset specificato
    """
    images = []
    image_info = []
    
    if not os.path.exists(dataset_path):
        print(f"Path non trovato: {dataset_path}")
        return images, image_info
    
    # Cerca ricorsivamente file immagine
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_images = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                all_images.append(os.path.join(root, file))
    
    # Seleziona campioni casuali
    if len(all_images) > num_samples:
        selected_images = random.sample(all_images, num_samples)
    else:
        selected_images = all_images
    
    for img_path in selected_images:
        try:
            img = Image.open(img_path)
            img = img.convert('RGB')
            images.append(img)
            
            # Estrai informazioni dal path
            rel_path = os.path.relpath(img_path, dataset_path)
            parts = rel_path.split(os.sep)
            artist = parts[0] if len(parts) > 0 else "Unknown"
            filename = os.path.basename(img_path)
            
            image_info.append({
                'artist': artist,
                'filename': filename,
                'path': img_path
            })
        except Exception as e:
            print(f"Errore nel caricare {img_path}: {e}")
    
    return images, image_info

def create_dataset_comparison_visualization():
    """
    Crea una visualizzazione che mostra la distinzione tra i due dataset
    """
    # Definisci i path dei dataset (modifica secondo la tua struttura)
    artgraph_path = "data/artgraph"
    artgraph_complementary_path = "data/artgraph_complementary"
    
    # Carica immagini campione
    artgraph_images, artgraph_info = load_sample_images(artgraph_path, num_samples=6)
    complementary_images, complementary_info = load_sample_images(artgraph_complementary_path, num_samples=6)
    
    # Crea la figura
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    fig.suptitle('ArtGraph Dataset Separation\nDistinction between Evaluation and Training Subsets', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Colori per distinguere i dataset
    artgraph_color = '#2E86AB'  # Blu
    complementary_color = '#A23B72'  # Rosa/Viola
    
    # Prima riga: Header per ArtGraph Few-Shot Subset
    for i in range(6):
        axes[0, i].text(0.5, 0.5, 'ArtGraph\nFew-Shot Subset\n(Evaluation)', 
                       ha='center', va='center', fontsize=12, fontweight='bold',
                       color=artgraph_color, transform=axes[0, i].transAxes)
        axes[0, i].set_facecolor('#E8F4F8')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        # Aggiungi bordo colorato
        for spine in axes[0, i].spines.values():
            spine.set_color(artgraph_color)
            spine.set_linewidth(3)
    
    # Seconda riga: Immagini ArtGraph
    for i in range(6):
        if i < len(artgraph_images):
            axes[1, i].imshow(artgraph_images[i])
            axes[1, i].set_title(f"{artgraph_info[i]['artist']}", 
                                fontsize=10, color=artgraph_color, fontweight='bold')
        else:
            axes[1, i].text(0.5, 0.5, 'No Image\nAvailable', 
                           ha='center', va='center', fontsize=10)
        
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        
        # Aggiungi bordo colorato
        for spine in axes[1, i].spines.values():
            spine.set_color(artgraph_color)
            spine.set_linewidth(2)
    
    # Terza riga: Header per ArtGraph Complementary
    for i in range(6):
        axes[2, i].text(0.5, 0.5, 'ArtGraph\nComplementary Subset\n(Training)', 
                       ha='center', va='center', fontsize=12, fontweight='bold',
                       color=complementary_color, transform=axes[2, i].transAxes)
        axes[2, i].set_facecolor('#F8E8F4')
        axes[2, i].set_xticks([])
        axes[2, i].set_yticks([])
        
        # Aggiungi bordo colorato
        for spine in axes[2, i].spines.values():
            spine.set_color(complementary_color)
            spine.set_linewidth(3)
    
    # Quarta riga: Immagini Complementary
    for i in range(6):
        if i < len(complementary_images):
            axes[3, i].imshow(complementary_images[i])
            axes[3, i].set_title(f"{complementary_info[i]['artist']}", 
                                fontsize=10, color=complementary_color, fontweight='bold')
        else:
            axes[3, i].text(0.5, 0.5, 'No Image\nAvailable', 
                           ha='center', va='center', fontsize=10)
        
        axes[3, i].set_xticks([])
        axes[3, i].set_yticks([])
        
        # Aggiungi bordo colorato
        for spine in axes[3, i].spines.values():
            spine.set_color(complementary_color)
            spine.set_linewidth(2)
    
    # Aggiungi freccia di separazione
    fig.text(0.02, 0.5, '↕\nSEPARATION\n↕', ha='center', va='center', 
             fontsize=14, fontweight='bold', rotation=0, color='red')
    
    # Aggiungi legenda informativa
    legend_text = """
    Key Distinctions:
    • Non-overlapping artist sets
    • Different usage purposes
    • No data leakage between sets
    • Complementary coverage of artistic styles
    """
    
    fig.text(0.98, 0.02, legend_text, ha='right', va='bottom', 
             fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, left=0.06, right=0.94)
    
    # Salva l'immagine
    output_path = "dataset_separation_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualizzazione salvata in: {output_path}")
    
    plt.show()

def print_dataset_statistics():
    """
    Stampa statistiche sui dataset per complementare la visualizzazione
    """
    print("="*60)
    print("ARTGRAPH DATASET SEPARATION ANALYSIS")
    print("="*60)
    
    artgraph_path = "data/artgraph"
    complementary_path = "data/artgraph_complementary"
    
    print(f"\n📊 Dataset Paths:")
    print(f"  • ArtGraph Few-Shot:    {artgraph_path}")
    print(f"  • ArtGraph Complementary: {complementary_path}")
    
    # Conta files in ciascun dataset
    def count_images_and_artists(path):
        if not os.path.exists(path):
            return 0, 0, []
        
        image_count = 0
        artists = set()
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for root, dirs, files in os.walk(path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_count += 1
                    # Estrai nome artista dalla struttura directory
                    rel_path = os.path.relpath(root, path)
                    artist = rel_path.split(os.sep)[0] if rel_path != '.' else "Unknown"
                    artists.add(artist)
        
        return image_count, len(artists), sorted(list(artists))
    
    ag_images, ag_artists_count, ag_artists = count_images_and_artists(artgraph_path)
    comp_images, comp_artists_count, comp_artists = count_images_and_artists(complementary_path)
    
    print(f"\n📈 Statistics:")
    print(f"  ArtGraph Few-Shot Subset:")
    print(f"    - Images: {ag_images}")
    print(f"    - Artists: {ag_artists_count}")
    
    print(f"  ArtGraph Complementary Subset:")
    print(f"    - Images: {comp_images}")
    print(f"    - Artists: {comp_artists_count}")
    
    # Controlla sovrapposizioni
    overlap = set(ag_artists) & set(comp_artists)
    print(f"\n🔍 Overlap Analysis:")
    print(f"  - Overlapping artists: {len(overlap)}")
    if overlap:
        print(f"  - Artists in both sets: {sorted(list(overlap))}")
    else:
        print("  ✅ No artist overlap detected - datasets are properly separated!")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Stampa statistiche sui dataset
    print_dataset_statistics()
    
    # Crea la visualizzazione
    print("\nCreating dataset comparison visualization...")
    create_dataset_comparison_visualization()
