import subprocess
import os

shots = [1, 2, 4, 8, 16]
seed = 42
config_file = "configs/artgraph.yaml"

# Imposta CUDA_VISIBLE_DEVICES se necessario, altrimenti puoi rimuovere questa riga
# se è già impostata globalmente o non ti serve specificarla qui.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Descommenta se vuoi impostarlo da script

for shot_number in shots:
    print(f"Running with shot_number = {shot_number}")
    command = [
        "python", "main.py",
        "--config", config_file,
        "--shot", str(shot_number),
        "--seed", str(seed)
    ]
    
    # Se hai descomentato la riga os.environ sopra, puoi omettere "CUDA_VISIBLE_DEVICES=0" qui
    # altrimenti, se vuoi che sia parte del comando specifico eseguito da subprocess:
    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        # Usa env=env_vars se hai bisogno di impostare CUDA_VISIBLE_DEVICES specificamente per questo comando
        process = subprocess.run(command, check=True, env=env_vars, text=True, capture_output=True)
        print(f"Output for shot {shot_number}:\n{process.stdout}")
        if process.stderr:
            print(f"Errors for shot {shot_number}:\n{process.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error running command for shot {shot_number}: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: main.py not found. Make sure you are in the correct directory.")
        break # Esce dal loop se il file principale non viene trovato
    print("-" * 30)

print("All runs completed.")