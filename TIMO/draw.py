import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
import os
from PIL import Image
import glob

# Crea la figura
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

# Definisci i colori per i dataset
colors = {
    'evaluation': '#87CEEB',  # Azzurro chiaro per Few-Shot Subset (Evaluation)
    'training': '#DDA0DD'     # Viola chiaro per Complementary Subset (Training)
}

# Percorsi delle cartelle delle immagini dai dataset ArtGraph
query_images_path = os.path.expandvars("/home/fonty/Scrivania/UniRepo/ComputerVision/TIMO/$DATA/artgraph")  # Few-Shot Subset (Evaluation)
support_images_path = os.path.expandvars("/home/fonty/Scrivania/UniRepo/ComputerVision/TIMO/$DATA/artgraph_complementary")  # Complementary Subset (Training)

# Funzione per caricare le immagini da una cartella
def load_images_from_folder(folder_path, max_images=None):
    images = []
    if not os.path.exists(folder_path):
        print(f"Cartella non trovata: {folder_path}")
        return images
    
    # Supporta vari formati di immagine
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    
    # Cerca le immagini nelle sottocartelle degli artisti
    images_folder = os.path.join(folder_path, 'images')
    if os.path.exists(images_folder):
        # Naviga attraverso le cartelle degli artisti
        for artist_folder in os.listdir(images_folder):
            artist_path = os.path.join(images_folder, artist_folder)
            if os.path.isdir(artist_path):
                for ext in extensions:
                    image_files.extend(glob.glob(os.path.join(artist_path, ext)))
                    image_files.extend(glob.glob(os.path.join(artist_path, ext.upper())))
    else:
        # Fallback: cerca direttamente nella cartella principale
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    print(f"Trovate {len(image_files)} immagini in {folder_path}")
    
    # Limita il numero di immagini se specificato
    if max_images:
        image_files = image_files[:max_images]
    
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            img = img.convert('RGB')  # Converte in RGB se necessario
            images.append(img)
        except Exception as e:
            print(f"Errore nel caricamento di {img_path}: {e}")
    
    return images

# Carica le immagini - 8 per entrambi i dataset
query_images = load_images_from_folder(query_images_path, max_images=8)
support_images = load_images_from_folder(support_images_path, max_images=8)

# Se non ci sono abbastanza immagini, crea dei placeholder
def create_placeholder_image():
    """Crea un'immagine placeholder grigia"""
    img = Image.new('RGB', (200, 200), color='lightgray')
    return img

# Assicurati di avere abbastanza immagini
while len(query_images) < 8:
    query_images.append(create_placeholder_image())

while len(support_images) < 8:
    support_images.append(create_placeholder_image())

# Titolo
ax.text(6, 5.5, 'ArtGraph Dataset Separation', fontsize=20, fontweight='bold', ha='center')

# Posizioni delle immagini - Query set (2x4) - centrate nel box
query_positions = [
    (1.4, 4.0), (1.9, 4.0),
    (1.4, 3.6), (1.9, 3.6),
    (1.4, 3.2), (1.9, 3.2),
    (1.4, 2.8), (1.9, 2.8)
]

# Support set (2x4) - centrate nel box
support_positions = [
    (3.9, 4.0), (4.4, 4.0),
    (3.9, 3.6), (4.4, 3.6),
    (3.9, 3.2), (4.4, 3.2),
    (3.9, 2.8), (4.4, 2.8)
]

# Disegna il box tratteggiato per Query set - larghezza aumentata
query_box = Rectangle((1.0, 2.6), 1.4, 1.7, linewidth=2, edgecolor=colors['evaluation'], 
                     facecolor='none', linestyle='--', alpha=0.8)
ax.add_patch(query_box)
ax.text(1.7, 4.5, 'Artgraph', fontsize=14, fontweight='bold', ha='center')

# Disegna le immagini del Query set
for i, (x, y) in enumerate(query_positions):
    if i < len(query_images):
        img_ax = fig.add_axes([x/12-0.025, y/6-0.025, 0.05, 0.05])
        img_ax.imshow(query_images[i])
        img_ax.set_xticks([])
        img_ax.set_yticks([])
        
        for spine in img_ax.spines.values():
            spine.set_edgecolor(colors['evaluation'])
            spine.set_linewidth(2)

# Disegna il box tratteggiato per Support set - larghezza aumentata
support_box = Rectangle((3.5, 2.6), 1.4, 1.7, linewidth=2, edgecolor=colors['training'], 
                       facecolor='none', linestyle='--', alpha=0.8)
ax.add_patch(support_box)
ax.text(4.2, 4.5, 'Artgraph_complementary', fontsize=14, fontweight='bold', ha='center')

# Disegna le immagini del Support set
for i, (x, y) in enumerate(support_positions):
    if i < len(support_images):
        img_ax = fig.add_axes([x/12-0.025, y/6-0.025, 0.05, 0.05])
        img_ax.imshow(support_images[i])
        img_ax.set_xticks([])
        img_ax.set_yticks([])
        
        for spine in img_ax.spines.values():
            spine.set_edgecolor(colors['training'])
            spine.set_linewidth(2)

# Leggenda spostata più a destra per bilanciare
legend_box = Rectangle((6.0, 2.8), 2.5, 1.4, linewidth=2, edgecolor='#8B4513', 
                      facecolor='#F5E6D3', linestyle='--', alpha=0.8)
ax.add_patch(legend_box)
ax.text(7.25, 3.9, 'DATASET LEGEND', fontsize=12, fontweight='bold', ha='center')

# Elementi della leggenda
legend_items = [
    (6.2, 3.5, colors['evaluation'], 'ArtGraph'),
    (6.2, 3.1, colors['training'], 'ArtGraph Complementary'),
]

for x, y, color, label in legend_items:
    # Box colorato per la leggenda
    legend_rect = Rectangle((x, y-0.08), 0.15, 0.15, facecolor=color, alpha=0.7, 
                           edgecolor=color, linewidth=2)
    ax.add_patch(legend_rect)
    ax.text(x + 0.25, y, label, va='center', fontsize=10)

plt.tight_layout()
plt.savefig('artgraph_dataset_separation.png', dpi=300, bbox_inches='tight')
plt.show()

# Salva la figura se necessario
