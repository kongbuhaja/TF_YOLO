import tensorflow as tf
from src.cfg import Config
from src.utils import Env
from src.data import Dataset
from src.models import Model

from src.engine.validator import Validator
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.engine.benchmarker import Benchmarker
from src.engine.exporter import Exporter

__all__ = (
    "Engine",
    "Trainer",
    "Validator",
    "Evaluator",
    "Exporter",
    "Benchmarker",
)


class Engine():
    def __init__(self, **kwargs):
        self.cfg = Config(**kwargs)
        if "epochs" not in kwargs:
            self.cfg._default_epochs = True
        self.env = Env(self.cfg.environment)
        self._set_amp_policy()
        self.model = Model(self.cfg.model)
        self.dataset = Dataset(self.cfg.data)

    def _set_amp_policy(self):
        amp = str(getattr(self.cfg, "amp", False))
        dtype_map = {"fp16": "float16", "bf16": "bfloat16"}
        if amp in dtype_map:
            try:
                tf.keras.mixed_precision.set_global_policy(f"mixed_{dtype_map[amp]}")
            except RuntimeError:
                pass

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

    def export(self, weights=None, format="tensorrt", dtype="fp16"):
        if weights is None:
            weights = self.model.weights_path / "best"
        if not hasattr(self, "exporter") or self.exporter.weights != weights:
            self.exporter = Exporter(self.model, weights,
                                     [1] + self.cfg.input_shape,
                                     self.model.normalize)
        return self.exporter.export(format, dtype=dtype)

    def benchmark(self, weights=None, format="tensorrt", dtype="fp16", gpu=0):
        if weights is None:
            weights = self.model.weights_path / "best"
        if not hasattr(self, "exporter") or self.exporter.weights != weights:
            self.exporter = Exporter(self.model, weights,
                                     [1] + self.cfg.input_shape, self.model.normalize)

        if self.exporter.is_exported(format, dtype):
            artifact_path = self.exporter.artifact_path(format, dtype)
            print(f"[Benchmark] Using existing export: {artifact_path}")
        else:
            print(f"[Benchmark] Exporting {format}/{dtype} ...")
            artifact_path = self.exporter.export(format, dtype=dtype)

        if not hasattr(self, "benchmarker"):
            self.benchmarker = Benchmarker(self.env, self.model, self.cfg, self.dataset)
        return self.benchmarker.benchmark(artifact_path, format, dtype, gpu)

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
