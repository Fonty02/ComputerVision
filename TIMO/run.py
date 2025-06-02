import subprocess
import os

shots = [1, 2, 4, 8, 16]
seed = 42
config_file = "configs/artgraph.yaml"



for shot_number in shots:
    print(f"Running with shot_number = {shot_number}")
    command = [
        "python", "main.py",
        "--config", config_file,
        "--shot", str(shot_number),
        "--seed", str(seed)
    ]
    

    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = "0"

    try:

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
        break 
    print("-" * 30)

print("All runs completed.")