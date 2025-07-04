from pathlib import Path
import yaml
import os

FILE = Path(__file__).resolve()
PATH = FILE.parents[0]
# CFG = FILE.parents[0]
DATASETS = PATH / "datasets"

# def yaml_load(file="data.yaml", path=DATASETS):
#     assert file in os.listdir(path), f'unknown {file}'
#     with open(path / file, encoding="utf-8") as f:
#         s = f.read()
#         cfg = yaml.safe_load(s)
#     if file != "default.yaml":
#         cfg.update(yaml_load("default.yaml", path.parents[0]))
#     cfg["name"] = file.split(".")[0]

#     return cfg

def yaml_load(file=PATH/"default.yaml"):
    assert os.path.exists(file), f'unknown {file}'
    with open(file, encoding="utf-8") as f:
        s = f.read()
        cfg = yaml.safe_load(s)
    return cfg

def dataset_yaml_load(file="coco.yaml"):
    return yaml_load(PATH / DATASETS / file)

def default_yaml_load(file="default.yaml"):
    return yaml_load(PATH / file)

class Config():
    def __init__(self, file=PATH/"default.yaml"):
        for key, value in yaml_load(file).items():
            setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)

# __all__ = (
#     default_yaml_load
#     dataset_yaml_load
# )