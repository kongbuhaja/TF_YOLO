from src.data import Dataloader

class Handler():
    def __init__(self, cfg, dataset, dtype):
        for name, value in cfg.__dict__.items():
            setattr(self, name, value)
        self.dataset = dataset
        self.dtype = dtype
        data = self.dataset.load(self.dtype, self.cache, self.workers)
        self.dataloader = Dataloader(cfg, data)
        self.category_map = data["category_map"]

    def __call__(self, model):
        pass