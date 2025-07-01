import json
import subprocess

with open("<config>.json", "r") as file:
    experiment = json.load(file)

modes = ["fit", "predict", "evaluate"]
algorithm_configs = [0]
folds = [0]

for data_set_name in experiment["data_set_names"]:
    for algorithm_name in experiment["algorithm_names"]:
        for algorithm_config in algorithm_configs:
            for fold in folds:
                for mode in modes:
                    try:
                        subprocess.run(
                            ["python", "execution_master.py", "--mode", mode, "--data_set_name", data_set_name,
                             "--algorithm_name", algorithm_name, "--algorithm_config", str(algorithm_config), "--fold",
                             str(fold)], timeout=3600)
                    except subprocess.TimeoutExpired:
                        print(f'Timeout expired')
