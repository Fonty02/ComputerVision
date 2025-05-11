import csv
import json
import os
from pathlib import Path
import re
from collections import Counter, defaultdict
import shutil


MIN_ARTWORKS_FOR_TRAIN_ABSOLUTE = 50 # Numero assoluto di opere per l'artista per essere incluso nel set di addestramento
MIN_ARTWORKS_FOR_TRAIN_PERCENTAGE = 0.03 # Percentuale minima di opere per l'artista rispetto al totale per essere incluso nel set di addestramento
MAX_ARTWORKS_FOR_TEST_ABSOLUTE = 10 # Numero massimo di opere per l'artista per essere incluso nel set di test
DEFAULT_SOURCE_IMAGES_SUBDIR = "images"

# Indici delle colonne nel CSV (0-based)
CSV_COL_IMAGE_FILENAME = 0
CSV_COL_ARTIST_NAME = 1
# --- End Configuration ---

def normalize_name(name):
    if not name:
        return "unknown"
    name = str(name).lower()
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^\w\s\(\)-]', '', name).strip()
    name = re.sub(r'\s+', '_', name)
    name = name.replace('(', '').replace(')', '')
    return name if name else "unknown"

def split_data_by_artist_artwork_count(csv_path, output_dir):
    print(f"Starting data split based on artist artwork count from: {csv_path}")
    if not csv_path.exists():
        print(f"Error: CSV file {csv_path} not found for splitting.")
        return False, None

    output_dir.mkdir(parents=True, exist_ok=True)

    artist_artworks_map = defaultdict(list) # artist_name -> [image_filename, ...]
    image_to_artist_name = {} # image_filename -> artist_name
    total_artworks_count = 0

    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header_skipped = False
        for i, row in enumerate(reader):
            if not header_skipped and row[CSV_COL_IMAGE_FILENAME].lower() == "image_file":
                print("Skipping header row in CSV.")
                header_skipped = True
                continue
            if len(row) <= max(CSV_COL_IMAGE_FILENAME, CSV_COL_ARTIST_NAME):
                print(f"Warning: Skipping malformed row {i+1} in {csv_path} (not enough columns for filename/artist): {row}")
                continue

            image_filename = row[CSV_COL_IMAGE_FILENAME].strip()
            artist_name = row[CSV_COL_ARTIST_NAME].strip()

            if not image_filename or not artist_name:
                print(f"Warning: Skipping row {i+1} due to missing image filename or artist name: {row}")
                continue
            
            artist_artworks_map[artist_name].append(image_filename)
            image_to_artist_name[image_filename] = artist_name
            total_artworks_count += 1

    if total_artworks_count == 0:
        print("Error: No artworks found in CSV. Splitting aborted.")
        return False, None
    
    print(f"Total artworks processed: {total_artworks_count}")
    print(f"Total unique artists found: {len(artist_artworks_map)}")

    sorted_unique_artists = sorted(list(artist_artworks_map.keys()))
    artist_to_id = {name: i for i, name in enumerate(sorted_unique_artists)}
    
    artist_to_id_path = output_dir / "artist_to_id.json"
    with open(artist_to_id_path, 'w', encoding='utf-8') as f:
        json.dump(artist_to_id, f, indent=4)
    print(f"Saved artist_to_id.json to {artist_to_id_path}")

    train_artists, test_artists, val_artists = set(), set(), set()

    for artist, artworks_list in artist_artworks_map.items():
        num_artworks = len(artworks_list)
        percentage_of_total = num_artworks / total_artworks_count
        if num_artworks >= MIN_ARTWORKS_FOR_TRAIN_ABSOLUTE or \
           percentage_of_total >= MIN_ARTWORKS_FOR_TRAIN_PERCENTAGE:
            train_artists.add(artist)

    for artist, artworks_list in artist_artworks_map.items():
        if artist in train_artists:
            continue
        if len(artworks_list) <= MAX_ARTWORKS_FOR_TEST_ABSOLUTE:
            test_artists.add(artist)

    for artist in artist_artworks_map.keys():
        if artist not in train_artists and artist not in test_artists:
            val_artists.add(artist)

    print(f"\nArtist distribution for splits:")
    print(f"Training artists ({len(train_artists)}): {sorted(list(train_artists))[:5]}...")
    print(f"Test artists ({len(test_artists)}): {sorted(list(test_artists))[:5]}...")
    print(f"Validation artists ({len(val_artists)}): {sorted(list(val_artists))[:5]}...")

    train_data, val_data, test_data = [], [], []

    for artist_name_key, image_files_list in artist_artworks_map.items():
        artist_id = artist_to_id[artist_name_key]
        for image_filename in image_files_list:
            item_for_individual_json = [image_filename, artist_name_key, artist_id]
            if artist_name_key in train_artists:
                train_data.append(item_for_individual_json)
            elif artist_name_key in test_artists:
                test_data.append(item_for_individual_json)
            elif artist_name_key in val_artists:
                val_data.append(item_for_individual_json)
    
    train_data.sort(key=lambda x: x[0])
    val_data.sort(key=lambda x: x[0])
    test_data.sort(key=lambda x: x[0])

    with open(output_dir / "train.json", 'w', encoding='utf-8') as f: json.dump(train_data, f, indent=4)
    with open(output_dir / "val.json", 'w', encoding='utf-8') as f: json.dump(val_data, f, indent=4)
    with open(output_dir / "test.json", 'w', encoding='utf-8') as f: json.dump(test_data, f, indent=4)

    print(f"\nArtwork distribution for splits (individual JSONs):")
    print(f"Training images: {len(train_data)}")
    print(f"Validation images: {len(val_data)}")
    print(f"Test images: {len(test_data)}")
    
    if not (train_data or val_data or test_data):
        print("Warning: No data was assigned to any split. Check your CSV and splitting criteria.")
        return False, None
    return True, image_to_artist_name 

def create_consolidated_split_json(output_dir, artist_to_id_path, image_to_artist_name_map):
    print(f"\nCreating consolidated artgraph_split.json in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).parent
    train_json_path = script_dir / "train.json"
    val_json_path = script_dir / "val.json"
    test_json_path = script_dir / "test.json"

    if not artist_to_id_path.exists():
        print(f"Error: {artist_to_id_path} not found. Cannot create consolidated JSON.")
        return False
    with open(artist_to_id_path, 'r', encoding='utf-8') as f:
        artist_to_id = json.load(f)

    consolidated_data = {"train": [], "val": [], "test": []}
    split_files = {"train": train_json_path, "val": val_json_path, "test": test_json_path}

    for split_name, json_file_path in split_files.items():
        if not json_file_path.exists():
            print(f"Warning: {json_file_path} not found. Skipping {split_name} for consolidated JSON.")
            continue
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data_for_split = json.load(f) 
        
        for item in data_for_split:
            image_filename, artist_name_from_item, artist_id_from_file = item[0], item[1], item[2]
            
            # L'artista per il percorso è artist_name_from_item
            normalized_artist_dir = normalize_name(artist_name_from_item)
            image_path_in_dtd_format = f"{normalized_artist_dir}/{image_filename}"
            
            consolidated_data[split_name].append([
                image_path_in_dtd_format,
                artist_id_from_file,
                artist_name_from_item
            ])
        consolidated_data[split_name].sort(key=lambda x: x[0])

    output_json_path = output_dir / "artgraph_split.json"
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(consolidated_data, f, indent=4)
    print(f"Successfully created {output_json_path}")
    return True

def create_artgraph_reorganized_labels_structure(csv_path, output_reorganized_dir):
    print(f"\nStarting data reorganization into DTD-like /labels structure at: {output_reorganized_dir}")
    labels_output_dir = output_reorganized_dir / "labels"
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    #
    image_details_for_labels = {} 

    print(f"Processing {csv_path} for reorganization metadata (artist names for paths and labels)...")
    if not csv_path.exists():
        print(f"Error: CSV file {csv_path} not found for reorganization.")
        return False

    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header_skipped = False
        for i, row in enumerate(reader):
            if not header_skipped and row[CSV_COL_IMAGE_FILENAME].lower() == "image_file":
                header_skipped = True
                continue
            if len(row) <= max(CSV_COL_IMAGE_FILENAME, CSV_COL_ARTIST_NAME):
                print(f"Warning: Skipping malformed row {i+1} in {csv_path} for labels: {row}")
                continue

            image_filename = row[CSV_COL_IMAGE_FILENAME].strip()
            artist_name_csv = row[CSV_COL_ARTIST_NAME].strip()

            if not image_filename or not artist_name_csv:
                continue

            normalized_artist_dir = normalize_name(artist_name_csv)
            image_path_for_txt = f"{normalized_artist_dir}/{image_filename}"
            

            label_for_joint = normalize_name(artist_name_csv) #
            
            image_details_for_labels[image_filename] = (normalized_artist_dir, image_path_for_txt, label_for_joint)

    joint_anno_file_path = labels_output_dir / "labels.txt"
    with open(joint_anno_file_path, 'w', encoding='utf-8') as joint_anno_file:
        sorted_filenames = sorted(image_details_for_labels.keys())
        for image_filename in sorted_filenames:
            _, image_path_for_txt, label_string = image_details_for_labels[image_filename]
            joint_anno_file.write(f"{image_path_for_txt} {label_string}\n")
    print(f"Created {joint_anno_file_path}")

    script_dir = Path(__file__).parent
    split_json_files = {
        "train": script_dir / "train.json",
        "val": script_dir / "val.json",
        "test": script_dir / "test.json",
    }

    for split_name, current_json_path in split_json_files.items():
        output_txt_path = labels_output_dir / f"{split_name}.txt"
        
        print(f"Processing {current_json_path} for {split_name}.txt split...")
        if not current_json_path.exists():
            print(f"Warning: JSON file {current_json_path} not found. Skipping {split_name}.txt.")
            continue
        try:
            with open(current_json_path, 'r', encoding='utf-8') as f_json:
                split_data_from_json = json.load(f_json)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {current_json_path}. Skipping {split_name}.txt.")
            continue
        
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            count_found, count_not_found = 0, 0
            sorted_split_data = sorted(split_data_from_json, key=lambda x: x[0])
            for item_data in sorted_split_data:
                image_filename_from_split_json = item_data[0].strip()
                if image_filename_from_split_json in image_details_for_labels:
                    _, image_path_for_txt, _ = image_details_for_labels[image_filename_from_split_json]
                    txt_file.write(f"{image_path_for_txt}\n")
                    count_found +=1
                else:
                    print(f"Warning: Image '{image_filename_from_split_json}' from {current_json_path} not found in CSV metadata cache for labels. Skipping for {split_name}.txt.")
                    count_not_found +=1
            print(f"Created {output_txt_path}. Images written: {count_found}, images not found/skipped: {count_not_found}.")
    print(f"Finished creating /labels structure in {labels_output_dir.resolve()}")
    return True

def copy_images_to_reorganized_structure(csv_path, source_images_base_dir, output_reorganized_images_dir):
    print(f"\nStarting image copy and reorganization by ARTIST...")
    print(f"Source images directory: {source_images_base_dir}")
    print(f"Target reorganized images directory: {output_reorganized_images_dir}")

    if not csv_path.exists():
        print(f"Error: CSV file {csv_path} not found. Cannot determine image artists for copying.")
        return False
    
    if not source_images_base_dir.exists() or not source_images_base_dir.is_dir():
        print(f"Error: Source images directory '{source_images_base_dir}' not found or is not a directory.")
        return False

    output_reorganized_images_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count, skipped_count, error_count = 0, 0, 0

    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header_skipped = False
        for i, row in enumerate(reader):
            if not header_skipped and row[CSV_COL_IMAGE_FILENAME].lower() == "image_file":
                print("Skipping header row in CSV for image copying.")
                header_skipped = True
                continue
            
            if len(row) <= max(CSV_COL_IMAGE_FILENAME, CSV_COL_ARTIST_NAME):
                print(f"Warning: Skipping malformed row {i+1} in {csv_path} during image copy (not enough columns for image/artist): {row}")
                skipped_count += 1
                continue

            image_filename = row[CSV_COL_IMAGE_FILENAME].strip()
            artist_name_csv = row[CSV_COL_ARTIST_NAME].strip() # Usiamo l'artista per la sottocartella

            if not image_filename or not artist_name_csv:
                print(f"Warning: Skipping row {i+1} due to missing image filename or artist name for copying: {row}")
                skipped_count += 1
                continue
            
            source_image_path = source_images_base_dir / image_filename
            if not source_image_path.exists():
                print(f"Warning: Source image '{source_image_path}' not found. Skipping copy.")
                skipped_count += 1
                continue

            normalized_artist_dir_name = normalize_name(artist_name_csv)
            
            target_artist_subdir = output_reorganized_images_dir / normalized_artist_dir_name
            target_artist_subdir.mkdir(parents=True, exist_ok=True)
            
            target_image_path = target_artist_subdir / image_filename
            
            try:
                if not target_image_path.exists():
                    shutil.copy2(source_image_path, target_image_path)
                    copied_count += 1
                else:
                    skipped_count +=1
            except Exception as e:
                print(f"Error copying '{source_image_path}' to '{target_image_path}': {e}")
                error_count += 1
    
    print(f"Image copying finished. Copied: {copied_count}, Skipped (missing source/already exists): {skipped_count}, Errors: {error_count}")
    return error_count == 0


if __name__ == "__main__":
    script_file_path = Path(__file__).resolve()
    artgraph_dir = script_file_path.parent
    project_root = artgraph_dir.parent

    csv_input_path = artgraph_dir / "artgraph.csv"
    split_json_output_dir = artgraph_dir 
    source_images_dir = artgraph_dir / DEFAULT_SOURCE_IMAGES_SUBDIR
    
    print("--- Step 1: Splitting data by artist artwork count ---")
    split_successful, image_to_artist_name_data = split_data_by_artist_artwork_count(
        csv_path=csv_input_path,
        output_dir=split_json_output_dir
    )

    if split_successful and image_to_artist_name_data:
        data_dir_base = project_root / "$DATA" 
        data_dir_base.mkdir(parents=True, exist_ok=True)
        output_reorganized_base_dir = data_dir_base / "artgraph"
        output_reorganized_base_dir.mkdir(parents=True, exist_ok=True)

        artist_to_id_file_path = split_json_output_dir / "artist_to_id.json"

        print("\n--- Step 2: Creating DTD-like consolidated artgraph_split.json ---")
        consolidated_json_successful = create_consolidated_split_json(
            output_dir=output_reorganized_base_dir,
            artist_to_id_path=artist_to_id_file_path,
            image_to_artist_name_map=image_to_artist_name_data 
        )

        print("\n--- Step 3: Creating DTD-like /labels structure ---")
        labels_structure_successful = create_artgraph_reorganized_labels_structure(
            csv_path=csv_input_path, 
            output_reorganized_dir=output_reorganized_base_dir
        )
        
        print("\n--- Step 4: Copying and reorganizing images into artgraph/images/ARTIST_NAME/ ---")
        output_reorganized_images_target_dir = output_reorganized_base_dir / "images"
        images_copied_successfully = copy_images_to_reorganized_structure(
            csv_path=csv_input_path, 
            source_images_base_dir=source_images_dir,
            output_reorganized_images_dir=output_reorganized_images_target_dir
        )
        
        if consolidated_json_successful and labels_structure_successful and images_copied_successfully:
            print("\nSuccessfully created DTD-like structure for ArtGraph (organized by ARTIST).")
            print(f"Main split file: {output_reorganized_base_dir / 'artgraph_split.json'}")
            print(f"Label files: {output_reorganized_base_dir / 'labels'}")
            print(f"Reorganized images: {output_reorganized_images_target_dir}")
        else:
            print("\nThere were issues creating parts of the DTD-like structure or copying images.")
    else:
        print("Skipping reorganization steps due to issues in initial data splitting or missing metadata.")

    print("\nScript finished.")
