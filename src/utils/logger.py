import tensorflow as tf
import numpy as np

class Logger:
    def __init__(self, split=None, keys=None):
        if split == "train":
            keys = ["Epoch", "GPU", "CPU", "Ins/Img", "Cls_Loss", "Box_Loss", "Dfl_Loss", "Lr"]
        elif split == "val":
            keys = ["Ins/Img", "GPU", "CPU", "mAP50", "mAP50:95", "Cls_Loss", "Box_Loss", "Dfl_Loss"]
        self.keys = keys
        self.data = {key: 0 for key in keys}

    def update(self, **data):
        for key, value in data.items():
            self.data[key] = self._to_scalar(value)

    def _to_scalar(self, value):
        if tf.is_tensor(value):
            value = value.numpy()

        if isinstance(value, (np.ndarray, np.generic)):
            value = value.item()
    
        return value
    
    def __getitem__(self, key):
        return self.data[key]
    
    def items(self):
        return self.data.items()
    
    def values(self):
        return self.data.values()
    
    