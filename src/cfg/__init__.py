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
        cfg = yaml.safe_load(f.read())
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
            self.data.path = Path(self.data.path).resolve()
            for key, value in self.data.dirs.items():
                self.data.dirs[key] = self.data.path / value

    def __repr__(self):
        text = ""
        for key, value in self.__dict__.items():
            text += f"self.{key}: "
            if isinstance(value, Config):
                text += "".join(f"\n    {line}" for line in str(value).split("\n"))
            elif isinstance(value, dict):
                text += "".join(f"\n    {k}: {v}" for k, v in value.items())
            else:
                text += f"{value}"
            text += "\n"
        return text[:-1]

__all__ = (
    "Config",
)