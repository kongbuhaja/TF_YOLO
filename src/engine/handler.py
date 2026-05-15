from src.data import Dataloader
from src.utils.logger import Logger
from pathlib import Path
from src.utils.progress import ProgressBar
import os

class Handler():
    def __init__(self, env, model, cfg, dataset, split):
        self.env = env
        self.model = model
        self.cfg = cfg
        self.dataset = dataset
        self.split = split
        self.logger = Logger(self.split)
        # data = self.dataset.load(self.split, self.cfg.cache, self.cfg.workers)
        data = self.dataset.load("val", self.cfg.cache, self.cfg.workers)
    
        self.dataloader = Dataloader(cfg, data)
        self.category_map = data["category_map"]
        self.pbar = ProgressBar(self.dataloader,
                                task=self.cfg.task.upper(),
                                split=self.split,
                                headers=self.logger.keys)

    def make_dir(self, name):
        path = Path.cwd().resolve() / "results"
        os.makedirs(path, exist_ok=True)
        n = sum([d.startswith(name) for d in os.listdir(path)])
        self.cfg.path = path / f"{name}{n}"
        os.makedirs(self.cfg.path, exist_ok=True)

    def on_epoch_start(self):
        # self.pbar = ProgressBar(self.dataloader,
        #                         task=self.cfg.task.upper(),
        #                         split=self.split,
        #                         headers=self.logger.keys)
        pass

    def on_epoch_end(self):
        pass