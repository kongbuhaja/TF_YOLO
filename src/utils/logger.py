import tensorflow as tf
import numpy as np

class Logger:
    def __init__(self, split=None):
        if split == "train":
            self.data = {"Epoch": "", "Ins/Img": "", "GPU": "", "CPU": "", "LR": ""}
        elif split == "val":
            self.data = {"Ins/Img": "", "GPU": "", "CPU": "", "mAP50": "", "mAP50:95": ""}
        else:
            self.data = {}
        self._fixed = list(self.data.keys())

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
