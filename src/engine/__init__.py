from pathlib import Path

from src.cfg import Config
from src.utils import Env
from src.data import Dataset

__all__ = (
    "Engine",
    "Trainer",
    "Validator",
    "Evaluator",
)


def __getattr__(name):
    if name == "Trainer":
        from src.engine.trainer import Trainer
        return Trainer
    if name == "Validator":
        from src.engine.validator import Validator
        return Validator
    if name == "Evaluator":
        from src.engine.evaluator import Evaluator
        return Evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class Engine():
    def __init__(self, **kwargs):
        self.cfg = Config(**kwargs)
        self.env = Env(self.cfg.environment)
        self.dataset = Dataset(self.cfg.data)
        self.model_build()

    @staticmethod
    def _set_policy(policy):
        import tensorflow as tf
        try:
            tf.keras.mixed_precision.set_global_policy(policy)
        except RuntimeError:
            pass

    def model_build(self, amp=False):
        from src.tools.util import ModelParser
        dtype_map = {"fp16": "float16", "bf16": "bfloat16"}
        policy = f"mixed_{dtype_map[amp]}" if amp in dtype_map else "float32"
        if getattr(self, "_model_policy", None) == policy and hasattr(self, "model"):
            return
        parser = ModelParser(self.cfg.model)
        if parser.detect_type() == "tf":
            self._set_policy(policy)
        self.model = parser()
        self._model_policy = policy

    def train(self):
        amp = str(getattr(self.cfg, "amp", False))
        self.model_build(amp)
        if not hasattr(self, "trainer"):
            self.trainer = self.generate_handler("trainer")
        try:
            return self.trainer.train()
        finally:
            self._set_policy("float32")

    def evaluate(self):
        self.model_build()
        if not hasattr(self, "evaluator"):
            self.evaluator = self.generate_handler("evaluator")
        self.evaluator.evaluate()

    def generate_handler(self, role):
        from src.engine.trainer import Trainer
        from src.engine.validator import Validator
        from src.engine.evaluator import Evaluator
        handler_map = {"trainer": Trainer,
                       "validator": Validator,
                       "evaluator": Evaluator}
        
        if role not in handler_map:
            raise ValueError(f"{role} is not supported")

        return handler_map[role](self.env, self.model, self.cfg, self.dataset)

    def _get_exporter(self, weights=None):
        from src.tools.exporter import Exporter
        weights = Path(weights) if weights is not None else self.model.weights_path / "best"
        if getattr(self, "_exporter", None) is None or self._exporter.weights != weights:
            self._exporter = Exporter(weights,
                                      [1] + list(self.cfg.input_shape),
                                      self.model.normalize)
        return self._exporter

    def export(self, weights=None, format="tensorrt", dtype="fp16"):
        try:
            return self._get_exporter(weights).export(format, dtype=dtype)
        except Exception as e:
            print(f"[Export] {format}/{dtype} failed: {e}")
            return None

    def benchmark(self, weights=None, format="tensorrt", dtype="fp16", gpu=0, export=True):
        try:
            from src.tools.util import ExportedModel, load_exported_model
            from src.engine.benchmarker import Benchmarker

            if isinstance(self.model, ExportedModel):
                if getattr(self.model, "gpu", gpu) != gpu:
                    self.model.load(gpu=gpu)
                runtime_model, source_model = self.model, None
            else:
                exporter = self._get_exporter(weights)
                if exporter.is_exported(format, dtype):
                    artifact_path = exporter.artifact_path(format, dtype)
                    print(f"[Benchmark] Using existing export: {artifact_path}")
                elif export:
                    print(f"[Benchmark] Exporting {format}/{dtype} ...")
                    artifact_path = exporter.export(format, dtype=dtype)
                    if artifact_path is None:
                        print(f"[Benchmark] {format}/{dtype} skipped (export failed)")
                        return None
                else:
                    print(f"[Benchmark] {format}/{dtype} skipped "
                          f"(no exported model at {exporter.artifact_path(format, dtype)}, export=False)")
                    return None
                runtime_model = load_exported_model(artifact_path, gpu=gpu)
                source_model = self.model

            if not hasattr(self, "_benchmarker"):
                self._benchmarker = Benchmarker(self.env, self.cfg, self.dataset)
            return self._benchmarker.run(runtime_model, source_model=source_model)
        except Exception as e:
            print(f"[Benchmark] {format}/{dtype} failed: {e}")
            return None

    def on_epoch_start(self):
        self.dataloader.on_epoch_start()

    def on_epoch_end(self):
        self.dataloader.on_epoch_end()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import gc
        for attr in ("trainer", "evaluator", "validator", "model", "dataset", "_model_policy"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except ImportError:
            pass
