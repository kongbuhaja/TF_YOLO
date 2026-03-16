from pathlib import Path
import yaml
import os
import re

FILE = Path(__file__).resolve()
PATH = FILE.parents[0]
DEFAULT = PATH / "default.yaml"
DATASETS = PATH / "datasets"
MODELS = PATH / "models"

def yaml_load(file=PATH/"default.yaml"):
    assert os.path.exists(file), f"Config error | unknown file {file}"
    with open(file, encoding="utf-8") as f:
        cfg = yaml.safe_load(f.read())
    return cfg

import copy
import os
import re
from pathlib import Path

class Config:
    def __init__(self, dict_source=None, **kwargs):
        config_data = {}
        
        if dict_source is not None:
            config_data = dict_source
        elif kwargs.get("weight", False):
            config_data = kwargs
        else:
            file = kwargs.get("file", DEFAULT)
            config_data = yaml_load(file)
            config_data.update(kwargs)

        for key, value in config_data.items():
            if isinstance(value, dict) and not any(isinstance(k, int) for k in value.keys()):
                setattr(self, key, Config(dict_source=value))
            else:
                setattr(self, key, value)
                
        if hasattr(self, "model") and isinstance(self.model, (str, Path)):
            file, scale, weight = _get_model_file(self.model)
            
            inp_shape = getattr(self, "input_shape", None)
            
            self.model = Config(
                name=file.stem,
                file=file, 
                scale=scale,
                weight=weight,
                input_shape=inp_shape
            )

        if hasattr(self, "data") and isinstance(self.data, (str, Path)):
            data_file = DATASETS / f"{self.data}.yaml"
            self.data = Config(file=data_file, name=self.data)

            if hasattr(self.data, "path"):
                self.data.path = Path(self.data.path).resolve()

            if hasattr(self.data, "dirs"):
                for key, value in self.data.dirs.items():
                    setattr(self.data.dirs, key, self.data.path / value)

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        text = ""
        for key, value in self.items():
            text += f"self.{key}: "
            if isinstance(value, Config):
                # 중첩된 Config 객체 출력 (들여쓰기 추가)
                text += "".join(f"\n    {line}" for line in str(value).split("\n"))
            elif isinstance(value, dict):
                text += "".join(f"\n    {k}: {v}" for k, v in value.items())
            else:
                text += f"{value}"
            text += "\n"
        return text[:-1]
    
    def items(self):
        return self.__dict__.items()

def _get_model_file(file_name):
    def _get_all_files(path):
        files = {}
        for file in os.listdir(path):
            if os.path.isdir(path / file):
                files.update(_get_all_files(path / file))
            elif os.path.isfile(path / file):
                files[file] = (path / file)
        return files
        
    file_path = Path(file_name)
    model_name, ext = file_path.stem, file_path.suffix

    if ext != ".yaml" and ext != "":
        return file_path, None, True

    files = _get_all_files(MODELS)
    
    pattern = re.compile(r"^(.*)([nsmlx])$|^(.+)$")
    match = pattern.match(model_name)
    name, scale = (match.group(1), match.group(2)) if match.group(1) is not None else (match.group(3), None)
    file = Path(name).with_suffix(".yaml").name

    if os.path.exists(file_path):
        return file_path, scale, False
    elif file in files:
        return files[file], scale, False
    else:
        raise FileNotFoundError(f"Model error | unknown model {file}")


__all__ = (
    "Config",
)