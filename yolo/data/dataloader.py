from tensorflow.keras.utils import Sequence
from yolo.data.dataset import Dataset
import numpy as np
from yolo.data.process import Process, Postprocess

class Dataloader(Sequence):
    def __init__(self, cfg, dtype="val"):
        assert dtype.lower() in ["train", "val", "test"], "only support [train, val, test]"
        self.dtype = dtype.lower()
        self.cfg = cfg
        self.data = Dataset(cfg.dataset).read_data(dtype)
        self.indices = np.arange(len(self.data))
        self.epochs = cfg.epochs
        self.epoch = 0
        self.close_mosaic = self.epochs - cfg.close_mosaic + 1
    
        self.preprocess = Process(self.cfg.preprocess)
        self.process = Postprocess(self.cfg.process)
        self.process.set(dtype)
        
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.cfg.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.cfg.batch_size : (idx + 1) * self.cfg.batch_size]
        batch_data = [self.data[i] for i in batch_indices]

        serialized_data = []
        for data in batch_data:
            data = self.preprocess(*data)
            serialized_data.append(data)
        data = self.process(serialized_data)
        return data
    
    def on_epoch_start(self):
        self.epoch += 1
        if self.dtype == "train":
            if self.cfg.shuffle:
                np.random.shuffle(self.data)
            if self.epoch == self.close_mosaic:
                self.process.close_mosaic()
    
    def on_epoch_end(self):
        pass
        
    