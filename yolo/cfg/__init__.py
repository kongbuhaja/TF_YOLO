from pathlib import Path
import yaml
import os

FILE = Path(__file__).resolve()
CFG = FILE.parents[0]
DATASETS = CFG / "datasets"

def yaml_load(file="data.yaml", path=DATASETS):
    assert file in os.listdir(path), f'unknown {file}'
    with open(path / file, encoding="utf-8") as f:
        s = f.read()
        cfg = yaml.safe_load(s)
    if file != "default.yaml":
        cfg.update(yaml_load("default.yaml", path.parents[0]))
    cfg["name"] = file.split(".")[0]

    return cfg