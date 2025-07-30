from src.data.dataset import Dataset
import numpy as np
from src.data.process import Process, Augmentation
from threading import Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from copy import deepcopy

class Dataloader():
    def __init__(self, cfg, data={"data":[], "dtype":""}):
        for name, value in cfg.__dict__.items():
            setattr(self, name, value)
        self.epoch = 0
        self.close_mosaic = self.epochs - self.close_mosaic + 1
        self.insert_data(data)
        
        # prefetch
        self.queue = Queue(maxsize=self.prefetch)

    def insert_data(self, data):
        self.dtype = data["dtype"]
        self.data = data["data"]
        self.indices = np.arange(len(self.data))
        
        # process
        self.preprocess = Process(self.preprocess)

        if self.dtype == "train" or self.dtype == "val":
            self.process = Augmentation(self.augmentation) + Process(self.process)
        elif self.dtype == "val":
            self.process = Process(self.process)
        else:
            self.batch_size = 1
            self.process = Process({"resize_padding_with_info": self.process["resize_padding"][:2],
                                    "unsqueeze_coords": self.process["unsqueeze_coords"],
                                    "segment_to_task": self.process["segment_to_task"]})

        self.postprocess = Process(self.postprocess)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __iter__(self):
        self.on_epoch_start()
        return self
    
    def __next__(self):
        if self.idx >= len(self):
            self.on_epoch_end()
            raise StopIteration
        data = self.queue.get()
        self.idx += 1
        return data

    def _prefetch(self):
        while not self._stop_signal and self.prefetch_idx < len(self):
            batch_indices = self.indices[self.prefetch_idx * self.batch_size : (self.prefetch_idx + 1) * self.batch_size]
            batch_data = [self.preprocess(self.data[i].copy()) for i in batch_indices]
            
            if getattr(self.process, "use_mosaic", False):
                batch_data = [[data.copy() for data in batch_data] for _ in range(len(batch_data))]

            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                data = list(executor.map(self.process, batch_data))
            data = self.postprocess(data)

            self.queue.put(data)
            self.prefetch_idx += 1

    def on_epoch_start(self):
        self.idx = 0
        self.prefetch_idx = 0
        self.epoch += 1
        
        if self.dtype == "train":
            if self.shuffle:
                np.random.shuffle(self.data)
            if self.epoch == self.close_mosaic:
                self.process.close_mosaic()

        self._stop_signal = False
        self.queue.queue.clear()
        self._prefetch_thread = Thread(target=self._prefetch)
        self._prefetch_thread.daemon = True
        self._prefetch_thread.start()
    
    def on_epoch_end(self):
        self._stop_signal = True
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join()
