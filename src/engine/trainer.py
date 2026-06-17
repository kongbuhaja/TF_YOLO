from src.engine.handler import Handler
from src.engine.validator import Validator
from src.utils.loss import DFLDetectionLoss
from src.utils.optimizer import Optimizer
from pathlib import Path
import numpy as np
import yaml
import tensorflow as tf

class Trainer(Handler):
    def __init__(self, env, model, cfg, dataset):
        super().__init__(env, model, cfg, dataset, "train")
        self.loss = DFLDetectionLoss(model, cfg.loss)
        self.steps_per_epoch = len(self.dataloader)
        self.global_step = 0
        self.total_steps = self.cfg.epochs * self.steps_per_epoch
        self.optimizer = Optimizer(cfg, self.total_steps, self.steps_per_epoch)
        self.monitor.set_output_path(self.model.path)

        self.validator = Validator(self.env, self.model, self.cfg, self.dataset) if self.cfg.period > 0 else None
        self.best_map = 0.0
        # chage the default value when load model

    def train(self, resume=False):
        if resume:
            pass
        try:
            for epoch in range(self.cfg.epochs):
                self._on_epoch_start()

                for data in self.pbar:
                    batch_image, batch_labels = data["image"], data["labels"]

                    self._on_iteration_start()

                    total_loss, loss_items = self._train_step(batch_image, batch_labels)

                    self._on_iteration_end(epoch, loss_items, batch_labels)

                self._on_epoch_end(epoch)

        except Exception as e:
            total_loss, loss_items = 0.0, {}
            print(f"Training loop interrupted: {e}")
            raise e

        return total_loss, loss_items

    @tf.function
    def _train_step(self, batch_image, batch_labels):
        with tf.GradientTape() as tape:
            raw_preds = self.model(batch_image, training=True)
            total_loss, loss_items = self.loss(raw_preds, batch_labels)

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.model.apply_gradients(zip(gradients, self.model.trainable_variables))

        return total_loss, loss_items

    def _on_epoch_start(self):
        super()._on_epoch_start()
        self.save_args()
        self.step = 0
        self.avg_loss_items = {}

    def _on_epoch_end(self, epoch):
        super()._on_epoch_end()

        train_losses = {k: float(v) for k, v in self.avg_loss_items.items()}
        val_losses = {}
        val_metrics = {}
        lr = float(self.optimizer.lr)

        if self.validator and epoch % self.cfg.period == 0:
            val_losses, val_metrics = self.validator.validate()

            current_map = val_metrics.get("mAP", 0.0)
            if self._save_checkpoint(epoch, self.best_map, current_map):
                self.best_map = current_map

        self.monitor.log_epoch(epoch, train_losses, val_losses, val_metrics, lr)

    def _on_iteration_start(self):
        self.optimizer.update(self.global_step)

    def _on_iteration_end(self, epoch, loss_items, batch_labels):
        for k, v in loss_items.items():
            v = float(v)
            if k in self.avg_loss_items:
                self.avg_loss_items[k] = (self.avg_loss_items[k] * self.step + v) / (self.step + 1)
            else:
                self.avg_loss_items[k] = v

        if self.step + 1 == len(self.pbar):
            loss_items = self.avg_loss_items
            
        def log_update():
            self.images += len(batch_labels)
            self.instances += np.sum(np.sum(batch_labels[..., 1:5], -1) > 0)

            log = {"Epoch": f"{epoch + 1}/{self.cfg.epochs}",
                   "Ins/Img": f"{self.instances}/{self.images}",
                   "LR": self.optimizer.lr,
                   **loss_items,
                   **self.env.get_info()}

            self.monitor.update(**log)

        self.env.update_info()
        log_update()
        self.pbar.set_status(**self.monitor.data)

        self.step += 1
        self.global_step += 1

    def _save_checkpoint(self, epoch, best_map, current_map):
        self.model.save_model(epoch)
        # self.save_model("last")
        if current_map > best_map:
            self.model.save_model("best")
            return True
        return False
    
    def save_args(self):
        def serialize(value, order=None, skip_keys=None):
            if isinstance(value, Path):
                return str(value)
            if hasattr(value, "items"):
                d = {}
                for k, v in value.items():
                    if skip_keys and k in skip_keys:
                        continue
                    d[k] = serialize(v)
                if order:
                    return {k: d[k] for k in order if k in d}
                return d
            if isinstance(value, (list, tuple)):
                return [serialize(v) for v in value]
            try:
                yaml.safe_dump(value)
                return value
            except:
                return str(value)
        args = {}
        skip_keys = {"path", "weights", "environment"}
        for key, value in self.cfg.items():
            if key in skip_keys:
                continue
            if key == "data":
                args[key] = serialize(value, order=["name", "path", "file", "dirs"], skip_keys={"classes", "urls"})
            elif key == "model":
                args[key] = serialize(value, skip_keys={"backbone", "head"})
            else:
                args[key] = serialize(value)

        keys = list(args.keys())
        if "name" in keys:
            keys.remove("name")
            keys.insert(0, "name")
        if "data" in keys:
            keys.remove("data")
            keys.append("data")
        args = {k: args[k] for k in keys}

        args_path = self.model.path / "args.yaml"
        with open(args_path, "w") as f:
            yaml.dump(args, f, Dumper=_FlowListDumper, default_flow_style=None, sort_keys=False)


class _FlowListDumper(yaml.SafeDumper):
    pass


def _flow_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def _block_dict(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=False)


_FlowListDumper.add_representer(list, _flow_list)
_FlowListDumper.add_representer(tuple, _flow_list)
_FlowListDumper.add_representer(dict, _block_dict)
