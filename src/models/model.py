import os
from pathlib import Path

class Model():
    def __init__(self, file):
        name, extension = file.split(".")
        self.name = name
        if extension == "yaml":
            self.make_dir(name)

    def __call__(self, data):
        pass

    def train(self):
        self.make_dir(self.name)
        pass

    def eval(self):
        self.make_dir("eval")
        pass

    def make_dir(self, name):
        path = Path.cwd().resolve() / "results"
        os.makedirs(path, exist_ok=True)
        n = sum([d.startswith(name) for d in os.listdir(path)])
        self.path = path / f"{name}{n}"
        os.makedirs(self.path)

class Empty_model():
    def __call__(self, data):
        return data