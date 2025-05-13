import google.generativeai as genai
import json
import os
import time
import random
from pathlib import Path
import re

# --- CONFIGURAZIONE ---
# Specifica qui i percorsi dei tuoi file
STYLE_GUIDE_JSON_PATH = Path("artgraph_style.json") # Percorso del file JSON con gli stili
ARTIST_LIST_TXT_PATH = Path("subdirectories.txt")   # Percorso del file TXT con gli artisti
OUTPUT_JSON_PATH = Path("artgraph.json") # Percorso del file JSON di output

# Ottieni la API Key (preferibilmente da variabile d'ambiente)
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("API Key non trovata. Imposta la variabile d'ambiente GOOGLE_API_KEY o inseriscila nello script.")

genai.configure(api_key=api_key)

MODEL_NAME = "gemini-2.0-flash-lite"

generation_config = {
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 1024,
  "response_mime_type": "text/plain",
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

NUM_SENTENCES = 1
API_CALL_DELAY = 2.05
MAX_RETRIES = 1

def load_json(file_path: Path) -> dict:
    """Carica dati da un file JSON."""
    try:
        with file_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Errore: File JSON non trovato a '{file_path}'")
        raise
    except json.JSONDecodeError:
        print(f"Errore: Formato JSON non valido in '{file_path}'")
        raise

def load_or_create_json(file_path: Path) -> dict:
    """Carica un file JSON se esiste, altrimenti crea un dizionario vuoto."""
    if file_path.exists():
        try:
            with file_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Errore: Formato JSON non valido in '{file_path}'. Creazione nuovo file.")
            return {}
    else:
        return {}

def save_json(data: dict, file_path: Path):
    """Salva i dati in un file JSON."""
    try:
        with file_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except IOError as e:
        print(f"Errore durante il salvataggio del file JSON: {e}")
        return False

def load_txt_lines(file_path: Path) -> list[str]:
    """Carica le linee da un file TXT, rimuovendo spazi bianchi."""
    try:
        with file_path.open('r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    except FileNotFoundError:
        print(f"Errore: File TXT non trovato a '{file_path}'")
        raise

def get_style_examples(style_data: dict, num_examples: int = 10) -> list[str]:
    """Estrae frasi esempio casuali dal JSON di stile."""
    all_sentences = []
    for style_sentences in style_data.values():
        all_sentences.extend(style_sentences)

    if not all_sentences:
        return []

    num_examples = min(num_examples, len(all_sentences))
    return random.sample(all_sentences, num_examples)

def clean_response_sentences(text: str) -> list[str]:
    """Pulisce la risposta dell'LLM e la divide in frasi."""
    if not text:
        return []
    
    # Rimuove le citazioni e altri caratteri di formattazione
    text = re.sub(r'^[\s"\']*|[\s"\']*$', '', text)
    text = re.sub(r'[""]', '', text)
    
    # Divide in frasi
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Pulisce e filtra le frasi vuote
    cleaned_sentences = []
    for sentence in sentences:
        s = sentence.strip()
        if s and not s.isdigit() and len(s) > 5:  # Ignora numeri isolati e frasi molto corte
            if not s.endswith(('.', '!', '?')):
                s += '.'
            cleaned_sentences.append(s)
    
    return cleaned_sentences

def generate_descriptions_for_artist(model, artist_name: str, style_examples: list[str]) -> list[str]:
    """Genera descrizioni per un artista usando l'LLM."""
    if not artist_name:
        return None

    for attempt in range(MAX_RETRIES):
        try:
            prompt = f"""
Generate {NUM_SENTENCES} descriptive sentences about the artist or artistic movement "{artist_name}".
Please respond DIRECTLY IN ENGLISH.

The sentences should describe their artistic style, technique, historical context, influence, or significance.
Make the descriptions informative yet concise, focusing on what makes this artist unique.

Here are some examples of the style and tone to use:
{' '.join(style_examples[:5])}

Your response should ONLY include the description sentences, without any introductions, numbering, or explanations.
If you are unable to generate a description, please respond with the name of the artist or movement only, without any additional text.
"""

            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            if response.text:
                return [response.text.strip()]
                
                print(f"    Avviso: Ricevute meno frasi del previsto ({len(cleaned_sentences)}/{NUM_SENTENCES}) per '{artist_name}'.")
                if attempt < MAX_RETRIES - 1:
                     print(f"    Attesa {API_CALL_DELAY * (attempt + 1)}s prima del prossimo tentativo...")
                     time.sleep(API_CALL_DELAY * (attempt + 1))
                     continue
                else:
                    return cleaned_sentences

        except Exception as e:
            print(f"  Errore durante la generazione per '{artist_name}' (Tentativo {attempt + 1}): {e}")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 print(f"  Bloccato per: {response.prompt_feedback.block_reason}")
                 return None
            if attempt < MAX_RETRIES - 1:
                wait_time = API_CALL_DELAY * (2 ** attempt)
                print(f"    Attesa {wait_time}s prima del prossimo tentativo...")
                time.sleep(wait_time)
            else:
                print(f"  Massimo numero di tentativi raggiunto per '{artist_name}'. Salto.")
                return None
    return None

def main():
    """Funzione principale dello script."""
    print("--- Avvio Generazione Descrizioni Artisti ---")

    # 1. Carica i dati
    print(f"Caricamento guida di stile da: {STYLE_GUIDE_JSON_PATH}")
    style_data = load_json(STYLE_GUIDE_JSON_PATH)
    style_examples = get_style_examples(style_data)
    if not style_examples:
        print("Attenzione: Nessuna frase esempio trovata nel file JSON di stile.")

    print(f"Caricamento lista artisti da: {ARTIST_LIST_TXT_PATH}")
    artist_names = load_txt_lines(ARTIST_LIST_TXT_PATH)
    print(f"Trovati {len(artist_names)} artisti da processare.")

    # 2. Carica o crea il file JSON di output
    print(f"Caricamento/creazione file di output: {OUTPUT_JSON_PATH}")
    artist_descriptions = load_or_create_json(OUTPUT_JSON_PATH)
    
    # 3. Inizializza modello LLM
    print(f"Inizializzazione modello LLM: {MODEL_NAME}")
    model = genai.GenerativeModel(MODEL_NAME)

    # 4. Processa gli artisti
    start_time = time.time()

    for i, artist_name in enumerate(artist_names):
        # Salta artisti già processati
        if artist_name in artist_descriptions:
            print(f"\n[{i + 1}/{len(artist_names)}] Saltando: {artist_name} (già processato)")
            continue
            
        print(f"\n[{i + 1}/{len(artist_names)}] Processando: {artist_name}")
        descriptions = generate_descriptions_for_artist(model, artist_name, style_examples)

        if descriptions:
            artist_descriptions[artist_name] = descriptions
            print(f"  -> Descrizioni generate per {artist_name}.")
        else:
            print(f"  -> Impossibile generare descrizioni per {artist_name} dopo {MAX_RETRIES} tentativi.")
            artist_descriptions[artist_name] = []
        
        # Salva il file dopo ogni artista
        print(f"  Salvataggio incrementale in: {OUTPUT_JSON_PATH}")
        save_json(artist_descriptions, OUTPUT_JSON_PATH)

        # Aggiungi un ritardo per rispettare i limiti API
        if i < len(artist_names) - 1:
             print(f"  Attesa {API_CALL_DELAY}s...")
             time.sleep(API_CALL_DELAY)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n--- Elaborazione Completata ---")
    print(f"Tempo totale: {total_time:.2f} secondi")
    generated_count = sum(1 for desc in artist_descriptions.values() if desc)
    failed_count = len(artist_names) - generated_count
    print(f"Artisti processati con successo: {generated_count}")
    print(f"Artisti falliti: {failed_count}")
    print("--- Script Terminato ---")

if __name__ == "__main__":
    main()
