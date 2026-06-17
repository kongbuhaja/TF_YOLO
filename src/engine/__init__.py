import tensorflow as tf
from src.cfg import Config
from src.utils import Env
from src.data import Dataset
from src.models import Model

from src.engine.validator import Validator
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer

__all__ = (
    "Engine",
    "Trainer",
    "Validator",
    "Evaluator",
)


class Engine():
    def __init__(self, **kwargs):
        self.cfg = Config(**kwargs)
        self.env = Env(self.cfg.environment)
        self.model = Model(self.cfg.model)
        self.dataset = Dataset(self.cfg.data)

    def train(self):
        if not hasattr(self, "trainer"):
            self.trainer = self.generate_handler("trainer")
        self.trainer.train()

    def evaluate(self):
        if not hasattr(self, "evaluator"):
            self.evaluator = self.generate_handler("evaluator")
        self.evaluator.evaluate()

    def generate_handler(self, role):
        handler_map = {"trainer": Trainer,
                       "validator": Validator,
                       "evaluator": Evaluator}
        
        if role not in handler_map:
            raise ValueError(f"{role} is not supported")

        return handler_map[role](self.env, self.model, self.cfg, self.dataset)

    def on_epoch_start(self):
        self.dataloader.on_epoch_start()

    def on_epoch_end(self):
        self.dataloader.on_epoch_end()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        tf.keras.backend.clear_session()

        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "trainer"):
            del self.trainer
        if hasattr(self, "evaluator"):
            del self.evaluator
        if hasattr(self, "dataset"):
            del self.dataset
