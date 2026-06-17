from src.data import Dataloader
from src.utils.monitor import Monitor
from src.utils.progress import ProgressBar


class Handler():
    def __init__(self, env, model, cfg, dataset, split):
        self.env = env
        self.model = model
        self.cfg = cfg
        self.dataset = dataset
        self.split = split
        self.monitor = Monitor(split)

        # this is for test.
        if split == "val" or split == "train":
            data = self.dataset.load("val", cfg.cache, cfg.workers)
        else:
            data = self.dataset.load(split, cfg.cache, cfg.workers)

        self.dataloader = Dataloader(cfg, data)
        self.category_map = data["category_map"]
        self.pbar = ProgressBar(self.dataloader,
                                task=cfg.task.upper(),
                                split=split,
                                headers=self.monitor.keys)

    def _on_epoch_start(self):
        self.images = 0
        self.instances = 0

    def _on_epoch_end(self):
        pass
