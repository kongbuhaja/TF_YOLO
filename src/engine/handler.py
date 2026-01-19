from src.data import Dataloader
from pathlib import Path
import os

class Handler():
    def __init__(self, model, cfg, dataset, dtype):
        for name, value in cfg.__dict__.items():
            setattr(self, name, value)
        self.dataset = dataset
        self.dtype = dtype
        data = self.dataset.load(self.dtype, self.cache, self.workers)
        self.model = model
        self.dataloader = Dataloader(cfg, data)
        self.category_map = data["category_map"]

    def __call__(self, model):
        pass

    def make_dir(self, name):
        path = Path.cwd().resolve() / "results"
        os.makedirs(path, exist_ok=True)
        n = sum([d.startswith(name) for d in os.listdir(path)])
        self.path = path / f"{name}{n}"
        os.makedirs(self.path, exist_ok=True)