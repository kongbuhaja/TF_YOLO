from pathlib import Path
import yaml
import os

FILE = Path(__file__).resolve()
PATH = FILE.parents[0]
DEFAULT = PATH / "default.yaml"
DATASETS = PATH / "datasets"
MODELS = PATH / "models"

def yaml_load(file=PATH/"default.yaml"):
    assert os.path.exists(file), f'Config error | unknown {file}'
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
            
        if hasattr(self, "model"):
            file, scale, weight_file = _get_model_file(self.model)
            self.model = Config(name=file.name.rstrip(".yaml"),
                                file=file, 
                                scale=scale,
                                weight_file=weight_file,
                                image_size=self.image_size)

        if hasattr(self, "data"):
            self.data = Config(file=DATASETS / f"{self.data}.yaml", 
                               name=self.data)
        
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

def _get_model_file(path):
    def _get_all_files(path):
        files = {}
        for file in os.listdir(path):
            if os.path.isdir(path / file):
                files.update(_get_all_files(path / file))
            elif os.path.isfile(path / file):
                files[file] = (path / file)
        return files

    def check_file(file):
        if os.path.exists(file):
            pass
        elif file.name in files:
            _file = file
            file = files[file.name]
            if _file.parent.as_posix() != ".":
                print(f"Config issue | Model file {_file} is not exist, so {file} is loaded.")
        else:
            file = None
        return file
        
    file = Path(path)
    name, ext = file.stem, file.suffix
    if ext != ".yaml":
        weight_file = file
        file = Path(name).with_suffix(".yaml")
    else:
        weight_file = None

    files = _get_all_files(MODELS)

    _file = check_file(file)
    if _file:
        file = _file
        scale = None
    else:
        name, scale = file.stem[:-1], file.stem[-1]
        file = file.with_stem(name)
        _file = check_file(file)
        if _file:
            file = _file
        else:
            assert False, f"Config error | Wrong model file: {path} ex) yolo11.yaml, yolo11.h5, /.../yolo11.yaml, /.../yolo11.h5"
            
    return file, scale, weight_file


__all__ = (
    "Config",
)