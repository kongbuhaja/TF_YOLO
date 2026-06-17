import os, csv
import tensorflow as tf
import numpy as np

class Monitor():
    def __init__(self, split=None, path=None):
        self.data = {}
        if split == "train":
            self.data = {"Epoch": "", "Ins/Img": "", "GPU": "", "CPU": "", "LR": ""}
        elif split == "val":
            self.data = {"Ins/Img": "", "GPU": "", "CPU": "", "mAP50": "", "mAP50:95": ""}
        elif split == "eval":
            self.data = {"images": "", "dets": "", "GPU": "", "CPU": ""}
        self._fixed = list(self.data.keys())

        self.rows = []
        self.headers = None

        if path is not None:
            self.set_output_path(path)

    def set_output_path(self, path):
        self.path = path
        self.csv_path = path / "results.csv"
        self.tb_path = path / "tensorboard"
        self.writer = tf.summary.create_file_writer(str(self.tb_path))

    def update(self, **data):
        for key, value in data.items():
            self.data[key] = self._to_scalar(value)

    def _to_scalar(self, value):
        if tf.is_tensor(value):
            value = value.numpy()
        if isinstance(value, (np.ndarray, np.generic)):
            value = value.item()
        return value

    @property
    def keys(self):
        return self._fixed + [k for k in self.data if k not in self._fixed]

    def __getitem__(self, key):
        return self.data[key]

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def log_epoch(self, epoch, train_losses, val_losses, metrics, lr):
        if not hasattr(self, "path"):
            return

        row = {}
        row["Epoch"] = epoch
        row.update({"metrics/precision": metrics.get("precision", ""),
                    "metrics/recall": metrics.get("recall", ""),
                    "metrics/mAP50": metrics.get("mAP50", ""),
                    "metrics/mAP": metrics.get("mAP", "")})
        row.update(self._prefix("train", train_losses))
        row.update(self._prefix("val", val_losses))
        row["lr"] = lr

        for k, v in row.items():
            if isinstance(v, (float, int)):
                row[k] = round(v, 6) if isinstance(v, float) else v

        self.rows.append(row)
        self._write_csv()
        self._write_tb(epoch, train_losses, val_losses, metrics, lr)

    def _prefix(self, prefix, losses):
        return {f"{prefix}/{k}": v for k, v in losses.items()}

    def _all_keys(self):
        keys = ["Epoch"]
        for row in self.rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        if "lr" not in keys:
            keys.append("lr")
        return keys

    def _write_csv(self):
        keys = self._all_keys()
        widths = {k: max(len(k), 8) for k in keys}
        for row in self.rows:
            for k, v in row.items():
                widths[k] = max(widths[k], len(str(v)))

        with open(self.csv_path, "w", newline="") as f:
            header = "".join(f"{k:<{widths[k] + 2}}" for k in keys if k in keys)
            f.write(header.rstrip() + "\n")

            for row in self.rows:
                line = "".join(f"{str(row.get(k, '')):<{widths[k] + 2}}" for k in keys)
                f.write(line.rstrip() + "\n")

    def _write_tb(self, epoch, train_losses, val_losses, metrics, lr):
        if not hasattr(self, "writer"):
            return
        if not hasattr(tf.summary, "scalar"):
            return
        with self.writer.as_default():
            for k, v in train_losses.items():
                tf.summary.scalar(f"train/{k}", v, step=epoch)
            for k, v in val_losses.items():
                tf.summary.scalar(f"val/{k}", v, step=epoch)
            for k, v in metrics.items():
                tf.summary.scalar(f"metrics/{k}", v, step=epoch)
            tf.summary.scalar("lr", lr, step=epoch)
        self.writer.flush()
