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
            file, scale, weight, saved_path = _get_model_file(self.model)
            
            input_shape = getattr(self, "input_shape", None)
            
            self.model = Config(
                name=file.stem,
                file=file, 
                scale=scale,
                weight=weight,
                input_shape=input_shape,
                saved_path=saved_path
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
                text += "".join(f"\n    {line}" for line in str(value).split("\n"))
            elif isinstance(value, dict):
                text += "".join(f"\n    {k}: {v}" for k, v in value.items())
            else:
                text += f"{value}"
            text += "\n"
        return text[:-1]
    
    def items(self):
        return self.__dict__.items()

    def __getitem__(self, key):
        return getattr(self, str(key))

def _get_all_files(path):
    files = {}
    for file in os.listdir(path):
        if os.path.isdir(path / file):
            files.update(_get_all_files(path / file))
        elif os.path.isfile(path / file):
            files[file] = (path / file)
    return files


def _get_model_file(file_name):
    def _is_saved_model_dir(path):
        return path.is_dir() and (path / "saved_model.pb").exists()
        
    file_path = Path(file_name).resolve()
    model_name, ext = file_path.stem, file_path.suffix

    if _is_saved_model_dir(file_path):
        return _resolve_saved_model(file_path)

    if ext != ".yaml" and ext != "":
        return file_path, None, True, file_path.parent

    files = _get_all_files(MODELS)
    
    pattern = re.compile(r"^(.*?)([nsmlx])?$")
    match = pattern.match(model_name)
    if match:
        name, scale = match.groups()
        if scale is None:
            scale = "n"
    file = Path(name).with_suffix(".yaml").name

    if os.path.exists(file_path):
        return file_path, None, False, None
    elif file in files:
        return files[file], scale, False, None
    else:
        raise FileNotFoundError(f"Model error | unknown model {file}")


def _resolve_saved_model(file_path):
    weights_dir = file_path.parent
    project_dir = weights_dir.parent
    dir_name = project_dir.name

    m = re.match(r"^(.+[nsmlx])(\d+)$", dir_name)
    if m:
        full_name = m.group(1)
    else:
        full_name = dir_name

    pattern = re.compile(r"^(.*)([nsmlx])$|^(.+)$")
    m2 = pattern.match(full_name)
    if m2.group(1) is not None:
        model_name = m2.group(1)
        scale = m2.group(2)
    else:
        model_name = m2.group(3)
        scale = None

    files = _get_all_files(MODELS)
    yaml_name = Path(model_name).with_suffix(".yaml").name
    if yaml_name not in files:
        raise FileNotFoundError(
            f"Model error | cannot find YAML '{yaml_name}' "
            f"(resolved from directory '{dir_name}')"
        )

    return files[yaml_name], scale, False, file_path


__all__ = (
    "Config",
)