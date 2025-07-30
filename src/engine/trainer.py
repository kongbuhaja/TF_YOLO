from src.engine.handler import Handler
from tqdm import tqdm

class Trainer(Handler):
    def __init__(self, cfg, dataset):
        super().__init__(cfg, dataset, "train")

    def __call__(self, model):
        for data in tqdm(self.dataloader,
                         total=len(self.dataloader),
                         desc=f"Train {self.dataset.name} data"):
            pass
        