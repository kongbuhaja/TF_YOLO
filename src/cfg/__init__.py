from pathlib import Path
import yaml
import os

FILE = Path(__file__).resolve()
PATH = FILE.parents[0]
DEFAULT = PATH /"default.yaml"
DATASETS = PATH / "datasets"

def yaml_load(file=PATH/"default.yaml"):
    assert os.path.exists(file), f'unknown {file}'
    with open(file, encoding="utf-8") as f:
        s = f.read()
        cfg = yaml.safe_load(s)
    return cfg

class Config():
    def __init__(self, **kwargs):
        file = kwargs.get("file", DEFAULT)
        for key, value in yaml_load(file).items():
            setattr(self, key, value)
        
        for key, value in kwargs.items():
            setattr(self, key, value)

        if hasattr(self, "data"):
            data_name = getattr(self, "data")
            self.data = Config(file = DATASETS / f"{data_name}.yaml")
            self.data.name = data_name

    def __repr__(self):
        return str(self.__dict__)

__all__ = (
    "Config",
)